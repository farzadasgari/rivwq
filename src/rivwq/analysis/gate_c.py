from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rivwq.analysis import overlap
from rivwq.gee import screening, session, width_cloud
from rivwq.usgs import coverage, pool, scout
from rivwq.usgs.flags import add_all_flags

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "gate_c.yaml"
TEXT = {"site_no": str}
GATE_VALUE = 42
RULE_COLUMNS = ["pass_width_10px", "pass_turbidity_cov", "pass_overlap", "pass_fchl_target"]
STRATUM_AXES = ["turbidity_class", "regulation_class", "regime_class", "cloud_class", "region"]
VARIABLE_LABELS = {
    "turbidity": "turb_cov",
    "dissolved_oxygen": "do_cov",
    "fchl": "fchl_cov",
    "discharge": "discharge_cov",
    "gage": "gage_cov",
}
OVERLAP_COLUMNS = [
    "legacy_do_waiver",
    "train_years_turbidity",
    "train_years_do",
    "train_years_fchl",
    "target_turbidity_ok",
    "target_do_ok",
    "target_fchl_ok",
    "pass_overlap_catalog",
    "train_years",
    "test_ok",
    "train_years_conservative",
    "test_ok_conservative",
    "pass_overlap_conservative",
    "pass_overlap_hydraulics",
    "coverage_estimated",
    "turbidity_estimated_years",
    "dissolved_oxygen_estimated_years",
]
TABLE_FRONT = [
    "station",
    "state",
    "drainage_area_sqmi",
    "role",
    "relaxed_pool",
    "width_m",
    "pixels_across",
    "width_m_occ25",
    "width_m_occ75",
    "water_share",
    "dates_per_year",
    "cs_share",
    "scl_share",
    "est_scenes_upper",
    "est_scenes_matched",
]
TURBIDITY_COLUMNS = [
    "turbidity_median",
    "turbidity_p95",
    "turbidity_max",
    "turbidity_share_above_200",
    "turbidity_gap_days",
    "turbidity_interior_gap_days",
    "turbidity_daily_last",
]
FLAG_COLUMNS = [
    "overlap_years_catalog",
    "fchl_years",
    "fchl_before_2025",
    "turbidity_only",
    "caveat",
    "caveat_reason",
    "lake_like",
    "gauge_near_water",
    "under_min_pixels_across",
    "turbidity_class",
    "regulation_class",
    "regime_class",
    "cloud_class",
    "turbidity_source",
    "dissolved_oxygen_source",
    "fchl_source",
]


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def region_of_state(state: str, regions: dict[str, list[str]]) -> str:
    for name, states in regions.items():
        if state in states:
            return name
    return "unknown"


def coverage_columns(years: list[int]) -> list[str]:
    return [f"{prefix}_{year}" for prefix in VARIABLE_LABELS.values() for year in years]


def rename_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    renames = {}
    for variable, label in VARIABLE_LABELS.items():
        for column in frame.columns:
            if column.startswith(f"{variable}_cov_"):
                renames[column] = column.replace(f"{variable}_cov_", f"{label}_")
    return frame.rename(columns=renames)


def estimate_scenes(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["est_scenes_upper"] = result["dates_total"] * result["cs_share"]
    result["est_scenes_matched"] = result["est_scenes_upper"] * result["turb_cov_mean"]
    return result


def add_rule_columns(table: pd.DataFrame, rules: dict) -> pd.DataFrame:
    result = table.copy()
    result["pass_width_10px"] = result["pixels_across"] >= rules["min_pixels_across"]
    result["pass_turbidity_cov"] = result["turb_cov_2025"] >= rules["min_turbidity_coverage_2025"]
    result["pass_fchl_target"] = result["fchl_before_2025"].astype(bool) & (
        result["fchl_years"] >= rules["min_fchl_years"]
    )
    return result


def build_gate_table(
    screened: pd.DataFrame,
    coverage_rows: pd.DataFrame,
    catalog_overlap: pd.Series,
    config: dict,
    strata: dict,
) -> pd.DataFrame:
    years = coverage.study_years(config["coverage"])
    result = screened.rename(
        columns={
            "station_nm": "station",
            "dates_per_year_mean": "dates_per_year",
            "cloud_score_share": "cs_share",
        }
    )
    result = result.join(rename_coverage(coverage_rows.set_index("site_no")))
    result["overlap_years_catalog"] = catalog_overlap.reindex(result.index)
    result["cloud_class"] = result["state"].apply(
        lambda state: scout.classify_cloudiness(state, strata)
    )
    result = estimate_scenes(result)
    result = add_rule_columns(result, config["rules"])
    result["pass_overlap_catalog"] = (
        result["overlap_years_catalog"] >= config["rules"]["min_overlap_years"]
    )
    result["legacy_do_waiver"] = (result["dissolved_oxygen_source"] == "no_series") | (
        result["fchl_source"] == "no_series"
    )
    result = overlap.add_overlap_columns(result, config["rules"])
    result = overlap.add_target_columns(result, config["rules"])
    result["region"] = result["state"].apply(
        lambda state: region_of_state(state, config["regions"])
    )
    ordered = TABLE_FRONT + coverage_columns(years)
    ordered += [name for name in coverage.STATISTIC_NAMES if name in result.columns]
    ordered += [name for name in TURBIDITY_COLUMNS if name not in coverage.STATISTIC_NAMES]
    ordered += FLAG_COLUMNS + ["region"] + RULE_COLUMNS + OVERLAP_COLUMNS
    ordered += ["turb_cov_mean", "do_cov_mean", "fchl_cov_mean", "dates_total"]
    ordered = list(dict.fromkeys(ordered))
    return result[[name for name in ordered if name in result.columns]].rename_axis("site")


def combined_counts(table: pd.DataFrame, overlap_column: str, label: str) -> list[dict]:
    core = table[["pass_width_10px", "pass_turbidity_cov", overlap_column]].all(axis=1)
    return [
        {"rule": f"width + turbidity coverage + {label}", "sites": int(core.sum())},
        {
            "rule": f"width + turbidity coverage + {label} + fchl target",
            "sites": int((core & table["pass_fchl_target"]).sum()),
        },
    ]


def rule_counts(table: pd.DataFrame) -> pd.DataFrame:
    rows = [{"rule": "screened sites", "sites": len(table)}]
    rows += [{"rule": name, "sites": int(table[name].sum())} for name in RULE_COLUMNS]
    for name in ["pass_overlap_conservative", "pass_overlap_hydraulics", "pass_overlap_catalog"]:
        rows.append({"rule": name, "sites": int(table[name].sum())})
    rows += combined_counts(table, "pass_overlap", "overlap")
    rows += combined_counts(table, "pass_overlap_conservative", "overlap conservative")
    return pd.DataFrame(rows)


def passing_core(table: pd.DataFrame) -> pd.DataFrame:
    mask = table[["pass_width_10px", "pass_turbidity_cov", "pass_overlap"]].all(axis=1)
    return table[mask]


def stratum_counts(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for axis in STRATUM_AXES:
        counts = table[axis].value_counts()
        for value, count in counts.items():
            fchl = int(table.loc[table[axis] == value, "pass_fchl_target"].sum())
            rows.append({"axis": axis, "stratum": value, "sites": int(count), "with_fchl": fchl})
    return pd.DataFrame(rows)


def best_wide_sites(table: pd.DataFrame, top: int) -> pd.DataFrame:
    wide = table[table["pass_width_10px"]]
    return wide.sort_values("est_scenes_matched", ascending=False).head(top)


def gap_limit_days(config: dict) -> float:
    return config["coverage"]["gap_months_limit"] * config["coverage"]["days_per_month"]


def gap_sites(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    limit = gap_limit_days(config)
    selected = table[table["turbidity_interior_gap_days"] > limit]
    return selected.sort_values("turbidity_interior_gap_days", ascending=False)


def edge_gap_sites(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    limit = gap_limit_days(config)
    edge = table["turbidity_gap_days"] > limit
    interior = table["turbidity_interior_gap_days"] > limit
    return table[edge & ~interior].sort_values("turbidity_gap_days", ascending=False)


def high_turbidity_sites(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    limit = config["rules"]["high_turbidity_share"]
    selected = table[table["turbidity_share_above_200"] > limit]
    return selected.sort_values("turbidity_share_above_200", ascending=False)


def format_cell(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return "" if np.isnan(value) else f"{value:.2f}"
    return str(value)


def markdown_table(frame: pd.DataFrame, index_name: str = "site") -> str:
    shown = frame.reset_index().rename(columns={frame.index.name or "index": index_name})
    header = "| " + " | ".join(shown.columns) + " |"
    rule = "| " + " | ".join("---" for _ in shown.columns) + " |"
    lines = [header, rule]
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(format_cell(value) for value in row) + " |")
    return "\n".join(lines)


def compact(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return table[columns]


def section(title: str, body: str) -> str:
    return f"## {title}\n\n{body}\n"


def overlap_summary(table: pd.DataFrame) -> str:
    changed = overlap.changed_by_conservative(table)
    catalog_changed = table["pass_overlap"] != table["pass_overlap_catalog"]
    columns = ["station", "state", "train_years", "train_years_conservative", "test_ok"]
    columns += ["test_ok_conservative", "turbidity_estimated_years", "pass_overlap"]
    columns += ["pass_overlap_conservative", "pass_width_10px", "pass_turbidity_cov"]
    estimated = int(table["coverage_estimated"].sum())
    lines = [
        f"- sites with estimated coverage in a rule series: {estimated}",
        f"- sites whose pass_overlap differs from the old catalog-based rule: "
        f"{int(catalog_changed.sum())}",
        f"- sites that change between pass_overlap and pass_overlap_conservative: {len(changed)}",
        "",
        markdown_table(changed[columns]),
    ]
    return "\n".join(lines)


def target_section(core: pd.DataFrame) -> str:
    only = overlap.turbidity_only_eligible(core)
    columns = ["station", "state", "dissolved_oxygen_source", "fchl_source", "train_years_do"]
    columns += ["train_years_fchl", "target_turbidity_ok", "target_do_ok", "target_fchl_ok"]
    return "\n".join(
        [
            f"Among the {len(core)} sites passing width, turbidity coverage and overlap.",
            "",
            markdown_table(overlap.target_summary(core), "no"),
            "",
            f"Eligible for turbidity only ({len(only)} sites):",
            "",
            markdown_table(only[columns]),
        ]
    )


def render_report(table: pd.DataFrame, pool_summary: dict, config: dict) -> str:
    core = passing_core(table)
    top = config["rules"]["top_sites"]
    best_columns = [
        "station",
        "state",
        "pixels_across",
        "dates_total",
        "cs_share",
        "turb_cov_mean",
        "est_scenes_upper",
        "est_scenes_matched",
        "pass_turbidity_cov",
        "pass_overlap",
        "pass_fchl_target",
    ]
    gap_columns = ["station", "state", "turbidity_interior_gap_days", "turbidity_gap_days"]
    gap_columns += ["turbidity_daily_last", "turb_cov_2025", "pass_turbidity_cov"]
    high_columns = ["station", "state", "turbidity_median", "turbidity_p95", "turbidity_max"]
    high_columns += ["turbidity_share_above_200"]
    sources = pd.DataFrame(
        {
            variable: table[f"{variable}_source"].value_counts()
            for variable in ["turbidity", "dissolved_oxygen", "fchl"]
        }
    ).fillna(0)
    parts = [
        f"# 04d: Gate C table, run {date.today().isoformat()}\n",
        "Screened sites only. No final set is chosen here; that is Gate C.\n",
        section(
            "Pool",
            "\n".join(f"- {name}: {value}" for name, value in pool_summary.items()),
        ),
        section("Counts per proposed rule", markdown_table(rule_counts(table), "no")),
        section("Overlap rule from real coverage", overlap_summary(table)),
        section("Per-target eligibility", target_section(core)),
        section(
            "Strata of sites passing width, turbidity coverage and overlap",
            f"{len(core)} sites.\n\n" + markdown_table(stratum_counts(core), "no"),
        ),
        section(
            f"Best {top} wide sites by est_scenes_matched (width rule only)",
            markdown_table(compact(best_wide_sites(table, top), best_columns)),
        ),
        section(
            "Sites with an interior turbidity gap longer than "
            f"{config['coverage']['gap_months_limit']} months (2018-2025)",
            markdown_table(compact(gap_sites(table, config), gap_columns)),
        ),
        section(
            "Sites whose only long gap is a late start or early end of the daily series",
            markdown_table(compact(edge_gap_sites(table, config), gap_columns)),
        ),
        section(
            "Sites with turbidity above "
            f"{config['coverage']['turbidity_high_fnu']:.0f} FNU on more than "
            f"{config['rules']['high_turbidity_share'] * 100:.0f} percent of days",
            markdown_table(compact(high_turbidity_sites(table, config), high_columns)),
        ),
        section("Data source per series (number of sites)", markdown_table(sources, "source")),
    ]
    return "\n".join(parts)


def write_outputs(table: pd.DataFrame, report: str, config: dict) -> list[Path]:
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = config["output_prefix"]
    csv_path = output_dir / f"{prefix}.csv"
    md_path = output_dir / f"{prefix}.md"
    table.to_csv(csv_path)
    md_path.write_text(report, encoding="utf-8")
    return [csv_path, md_path]


def read_text_codes(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns
    dtype = {name: str for name in columns if name == "site_no" or name.endswith("_code")}
    return pd.read_csv(path, dtype=dtype).set_index("site_no")


def read_tables(config: dict) -> dict[str, pd.DataFrame]:
    input_dir = Path(config["input_dir"]).expanduser()
    names = {
        "done": config["done_table_file"],
        "candidates": config["candidates_file"],
        "screened": config["screened_file"],
    }
    return {key: read_text_codes(input_dir / name) for key, name in names.items()}


def relaxed_site_rows(
    screened: pd.DataFrame, sites: list[str], scout_config: dict, cache: Path
) -> pd.DataFrame:
    table = screened.loc[sites].copy()
    partial_summaries = cache / "relaxed_summaries_partial.csv"
    partial_coverage = cache / "relaxed_coverage_partial.csv"
    cache.mkdir(parents=True, exist_ok=True)
    summaries = scout.fetch_site_summaries(table, scout_config, partial_summaries)
    sampled = scout.fetch_coverage(table, scout_config, partial_coverage)
    extra = summaries.set_index("site_no").join(sampled.set_index("site_no"))
    with_strata = scout.add_strata(
        table.drop(columns=extra.columns, errors="ignore"), extra, scout_config["strata"]
    )
    return add_all_flags(with_strata, scout_config)


def new_site_frame(
    tables: dict[str, pd.DataFrame], new_sites: list[str], scout_config: dict, cache: Path
) -> pd.DataFrame:
    candidates, screened = tables["candidates"], tables["screened"]
    in_candidates = [site for site in new_sites if site in candidates.index]
    outside = [site for site in new_sites if site not in candidates.index]
    frames = [candidates.loc[in_candidates]]
    if outside:
        frames.append(relaxed_site_rows(screened, outside, scout_config, cache))
    combined = pd.concat(frames).loc[new_sites]
    combined = combined[screening.BASE_COLUMNS].copy()
    combined.insert(0, "role", "pool")
    return combined


def main_series_codes(row: pd.Series, settings: dict) -> dict[str, str | None]:
    return {
        "turbidity": settings["turbidity_code"],
        "dissolved_oxygen": settings["dissolved_oxygen_code"]
        if pd.notna(row["dissolved_oxygen_code"])
        else None,
        "fchl": pool.normalise_code(row["chlorophyll_fluorescence_code"]),
    }


def hydraulic_series_codes(row: pd.Series, settings: dict) -> dict[str, str | None]:
    return {
        "discharge": settings["discharge_code"] if pd.notna(row["discharge_code"]) else None,
        "gage": settings["gage_code"] if pd.notna(row["gage_height_code"]) else None,
    }


def coverage_jobs(code_map: dict[str, dict[str, str | None]]) -> list[tuple[str, str]]:
    jobs = {(site, code) for site, codes in code_map.items() for code in codes.values() if code}
    return sorted(jobs)


def coverage_table(
    code_map: dict[str, dict[str, str | None]], settings: dict, cache: Path
) -> tuple[pd.DataFrame, dict]:
    jobs = coverage_jobs(code_map)
    outcomes = coverage.fetch_all(jobs, settings, cache)
    failed = [job for job, outcome in outcomes.items() if outcome == "failed"]
    if jobs and len(failed) / len(jobs) > settings["max_failed_share"]:
        raise SystemExit(f"NWIS unreachable: {len(failed)} of {len(jobs)} requests failed")
    rows = [
        coverage.site_coverage_row(site, codes, settings, cache) for site, codes in code_map.items()
    ]
    return pd.DataFrame(rows), {"jobs": len(jobs), "failed": failed}


def hydraulic_sites(screened: pd.DataFrame, coverage_rows: pd.DataFrame, rules: dict) -> list[str]:
    turbidity = coverage_rows.set_index("site_no")["turbidity_cov_2025"]
    wide = screened["pixels_across"] >= rules["min_pixels_across"]
    covered = turbidity.reindex(screened.index) >= rules["min_turbidity_coverage_2025"]
    return list(screened.index[wide & covered.fillna(False)])


def run_screening(
    width_config: dict, sites: pd.DataFrame, batch_size: int
) -> tuple[pd.DataFrame, dict]:
    cache = width_cloud.REPO_ROOT / width_config["cache_dir"]
    checked = width_cloud.verify_datasets(width_config)
    raw = width_cloud.fetch_raw_width(width_config, sites, cache, batch_size)
    first_year = int(width_config["cloud"]["start"][:4])
    last_year = int(width_config["cloud"]["end_exclusive"][:4]) - 1
    years = list(range(first_year, last_year + 1))
    per_date = width_cloud.fetch_cloud(width_config, sites, years, cache)
    width = screening.width_table(raw, width_config["width"])
    cloud = screening.cloud_summary(
        per_date, list(sites.index), years, width_config["cloud"]["min_valid_roi_fraction"]
    )
    return screening.assemble(sites, width, cloud), checked


def check_gate() -> None:
    import ee

    value = ee.Number(21).multiply(2).getInfo()
    print(value)
    if value != GATE_VALUE:
        raise SystemExit(f"Earth Engine gate failed: {value}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    width_config = width_cloud.load_config()
    scout_config = scout.load_config()
    session.initialize()
    check_gate()
    tables = read_tables(config)
    done_table = tables["done"]
    required = pool.required_variables(scout_config)
    pool_table = pool.build_pool(
        tables["candidates"],
        tables["screened"],
        set(done_table.index),
        config["pool"],
        required,
    )
    new_sites = list(pool_table.index[~pool_table["already_done"]])
    print(
        f"pool {len(pool_table)} new {len(new_sites)} relaxed {int(pool_table.relaxed_pool.sum())}"
    )
    cache = REPO_ROOT / config["cache_dir"]
    new_frame = new_site_frame(tables, new_sites, scout_config, cache)
    new_screened, checked = run_screening(width_config, new_frame, config["width_batch_size"])
    all_screened = pd.concat([done_table, new_screened])
    relaxed_all = pool.relaxed_sites(
        tables["screened"], config["pool"]["relaxed_missing_series"], required
    )
    all_screened["relaxed_pool"] = all_screened.index.isin(relaxed_all)
    width_cloud.write_manifest(
        width_config,
        checked,
        len(all_screened),
        Path(config["output_dir"]).expanduser() / config["manifest_name"],
    )
    sources = tables["screened"].loc[all_screened.index]
    settings = config["coverage"]
    main_codes = {site: main_series_codes(row, settings) for site, row in sources.iterrows()}
    coverage_rows, coverage_info = coverage_table(main_codes, settings, cache)
    wide_sites = hydraulic_sites(all_screened, coverage_rows, config["rules"])
    hydraulic_codes = {
        site: hydraulic_series_codes(sources.loc[site], settings) for site in wide_sites
    }
    hydraulic_rows, hydraulic_info = coverage_table(hydraulic_codes, settings, cache)
    coverage_info["jobs"] += hydraulic_info["jobs"]
    coverage_info["failed"] += hydraulic_info["failed"]
    coverage_rows = coverage_rows.merge(hydraulic_rows, on="site_no", how="left")
    catalog_overlap = tables["screened"]["overlap_years"]
    table = build_gate_table(
        all_screened, coverage_rows, catalog_overlap, config, scout_config["strata"]
    )
    pool_summary = {
        "pool size (candidates >= 1000 mi2 plus relaxed)": len(pool_table),
        "relaxed pool (no DO series)": int(pool_table["relaxed_pool"].sum()),
        "already screened in 04c": int(pool_table["already_done"].sum()),
        "new in this run": len(new_sites),
        "screened sites in table (04c plus new)": len(table),
        "sites with discharge and gage height coverage fetched": len(wide_sites),
        "NWIS series requests": coverage_info["jobs"],
        "NWIS requests failed": len(coverage_info["failed"]),
    }
    for path in write_outputs(table, render_report(table, pool_summary, config), config):
        print(path)
