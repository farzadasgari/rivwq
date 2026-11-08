from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rivwq.analysis import gate_c, overlap, sites
from rivwq.usgs import qc_rules

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "site_rules_iv.yaml"
SUPPORTED_COVERAGE = {
    "year_basis": "calendar_utc",
    "interval_statistic": "median",
    "merge_sensors": "union",
}
FLAG_COLUMNS = [
    "pass_turbidity_cov",
    "pass_overlap",
    "pass_overlap_conservative",
    "target_turbidity_ok",
    "target_do_ok",
    "target_fchl_ok",
    "coverage_estimated",
]


class SiteRulesError(RuntimeError):
    pass


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key, supported in SUPPORTED_COVERAGE.items():
        if config["coverage"][key] != supported:
            raise SiteRulesError(f"coverage.{key} must be {supported}")
    return config


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def study_years(settings: dict) -> list[int]:
    return list(range(settings["first_year"], settings["last_year"] + 1))


def year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(year=year, month=1, day=1, tz="UTC"), pd.Timestamp(
        year=year + 1, month=1, day=1, tz="UTC"
    )


def expected_slot_count(year: int, interval_seconds: float) -> int:
    start, end = year_bounds(year)
    return math.ceil((end - start).total_seconds() / interval_seconds)


def slot_ids(times: pd.Series, year: int, interval_seconds: float) -> np.ndarray:
    start, end = year_bounds(year)
    inside = times[(times >= start) & (times < end)]
    offsets = (inside - start).dt.total_seconds().to_numpy()
    return np.unique(np.floor(offsets / interval_seconds))


def sensor_intervals(tidy: pd.DataFrame, code: str) -> dict[str, float]:
    in_code = tidy[tidy["parameter_code"] == code]
    return {
        sensor: qc_rules.median_interval_seconds(group["datetime_utc"])
        for sensor, group in in_code.groupby("sensor_id")
    }


def parameter_coverage(
    flags: pd.DataFrame, tidy: pd.DataFrame, code: str, years: list[int], cap: float
) -> pd.DataFrame:
    intervals = sensor_intervals(tidy, code)
    columns = ["year", "coverage", "slots_covered", "slots_expected", "overlap_slots"]
    if not intervals:
        return pd.DataFrame(columns=columns + ["interval_minutes", "sensors"])
    finite = [value for value in intervals.values() if not math.isnan(value)]
    interval = min(finite)
    kept = flags[(flags["parameter_code"] == code) & ~flags["exclude"]]
    rows = []
    for year in years:
        per_sensor = [
            slot_ids(group["datetime_utc"], year, interval)
            for _, group in kept.groupby("sensor_id")
        ]
        union = np.unique(np.concatenate(per_sensor)) if per_sensor else np.array([])
        total = sum(len(ids) for ids in per_sensor)
        expected = expected_slot_count(year, interval)
        rows.append(
            {
                "year": year,
                "coverage": min(len(union) / expected, cap),
                "slots_covered": len(union),
                "slots_expected": expected,
                "overlap_slots": total - len(union),
            }
        )
    result = pd.DataFrame(rows)
    result["interval_minutes"] = interval / 60.0
    result["sensors"] = len(intervals)
    return result


def site_series_codes(record: dict | None, config: dict) -> dict[str, str | None]:
    codes: dict[str, str | None] = dict(config["series_codes"])
    codes["fchl"] = record["fchl_code"] if record else None
    return codes


def site_coverage(
    site: str, tidy: pd.DataFrame, codes: dict[str, str | None], config: dict
) -> pd.DataFrame:
    wanted = [code for code in codes.values() if code]
    subset = tidy[tidy["parameter_code"].isin(wanted)].reset_index(drop=True)
    qc_config = qc_rules.load_config(REPO_ROOT / config["qc_config"])
    flags = qc_rules.apply_level(subset, qc_config, config["qc_level"])
    years = study_years(config["coverage"])
    parts = []
    for target, code in codes.items():
        if code is None:
            continue
        part = parameter_coverage(flags, subset, code, years, config["coverage"]["cap"])
        part.insert(0, "target", target)
        part.insert(1, "code", code)
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    result.insert(0, "site_no", site)
    return result


def wide_coverage(long: pd.DataFrame, all_sites: list[str], years: list[int]) -> pd.DataFrame:
    wide = pd.DataFrame(index=pd.Index(all_sites, name="site_no"))
    for prefix in overlap.SERIES.values():
        for year in years:
            wide[f"{prefix}_{year}"] = 0.0
    for row in long.itertuples():
        wide.loc[row.site_no, f"{overlap.SERIES[row.target]}_{row.year}"] = row.coverage
    return wide


def apply_rules(old: pd.DataFrame, wide: pd.DataFrame, rules: dict) -> pd.DataFrame:
    table = old.copy()
    for column in wide.columns:
        table[column] = wide[column].reindex(table.index)
    table["turbidity_estimated_years"] = np.nan
    table["dissolved_oxygen_estimated_years"] = np.nan
    table = gate_c.add_rule_columns(table, rules)
    table = overlap.add_overlap_columns(table, rules)
    return overlap.add_target_columns(table, rules)


def refresh_scene_estimates(table: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    result = table.copy()
    result["turb_cov_mean"] = result[[f"turb_cov_{year}" for year in years]].mean(axis=1)
    return gate_c.estimate_scenes(result)


def frozen_table(old_full: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    columns = list(old_full.columns)
    missing = [column for column in columns if column not in new.columns]
    if missing:
        raise SiteRulesError(f"recomputed table lacks columns: {missing}")
    rest = old_full.drop(index=new.index)
    return pd.concat([rest, new[columns]]).loc[old_full.index]


def layer_label(table: pd.DataFrame) -> pd.Series:
    label = pd.Series("none", index=table.index)
    for name, mask in sites.layer_masks(table).items():
        label[mask] = name
    return label


def compare_flags(old: pd.DataFrame, new: pd.DataFrame, members: dict[str, str]) -> pd.DataFrame:
    index = pd.Index(list(members), name="site_no")
    result = pd.DataFrame({"frozen_layer": pd.Series(members)}, index=index)
    result["station"] = old.loc[index, "station"]
    old_layer = layer_label(old)
    new_layer = layer_label(new)
    for column in FLAG_COLUMNS:
        result[f"old_{column}"] = old.loc[index, column].astype(bool)
        result[f"new_{column}"] = new.loc[index, column].astype(bool)
    result["old_layer_by_rule"] = old_layer.loc[index]
    result["new_layer_by_rule"] = new_layer.loc[index]
    flag_changes = [result[f"old_{column}"] != result[f"new_{column}"] for column in FLAG_COLUMNS]
    changed = pd.concat(flag_changes, axis=1).any(axis=1)
    result["changed"] = changed | (result["old_layer_by_rule"] != result["new_layer_by_rule"])
    return result


def near_threshold_rows(
    long: pd.DataFrame, table: pd.DataFrame, rules: dict, margin: float
) -> pd.DataFrame:
    rows = []
    for row in long.itertuples():
        if row.target not in ("turbidity", "dissolved_oxygen", "fchl"):
            continue
        if row.year < rules["overlap_train_first_year"] or row.year > rules["overlap_test_year"]:
            continue
        if row.target == "fchl" and not table.loc[row.site_no, "target_fchl_ok"]:
            continue
        is_test = row.year == rules["overlap_test_year"]
        threshold = rules["min_test_coverage"] if is_test else rules["min_train_year_coverage"]
        distance = row.coverage - threshold
        if abs(distance) <= margin:
            rows.append(
                {
                    "site_no": row.site_no,
                    "target": row.target,
                    "year": row.year,
                    "coverage": row.coverage,
                    "threshold": threshold,
                    "distance": distance,
                    "passes_year": distance >= 0,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "site_no",
            "target",
            "year",
            "coverage",
            "threshold",
            "distance",
            "passes_year",
        ],
    )


def read_members(sites_file: Path) -> dict[str, dict]:
    with open(sites_file, encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    members = {}
    for layer in sites.LAYER_ORDER:
        for record in document["layers"][layer]:
            members[record["site_no"]] = {**record, "layer": layer}
    return members


def coverage_for_all(config: dict, members: dict[str, dict]) -> pd.DataFrame:
    run_dir = REPO_ROOT / config["raw_root"] / config["run_id"]
    parts = []
    for site, record in members.items():
        tidy = qc_rules.read_site(run_dir, site)
        codes = site_series_codes(record, config)
        parts.append(site_coverage(site, tidy, codes, config))
    return pd.concat(parts, ignore_index=True)


def write_outputs(frames: dict[str, pd.DataFrame], config: dict) -> list[Path]:
    target_dir = Path(config["output_dir"]).expanduser()
    written = []
    for name, frame in frames.items():
        path = target_dir / f"{config['output_prefix']}_{name}.csv"
        frame.to_csv(path, index=True if frame.index.name else False)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    gate_config = gate_c.load_config(REPO_ROOT / config["gate_config"])
    members = read_members(REPO_ROOT / config["sites_file"])
    old_full = sites.read_table(REPO_ROOT / config["old_table"])
    old = old_full
    long = coverage_for_all(config, members)
    wide = wide_coverage(long, list(members), study_years(config["coverage"]))
    years = study_years(config["coverage"])
    new = refresh_scene_estimates(
        apply_rules(old.loc[list(members)], wide, gate_config["rules"]), years
    )
    comparison = compare_flags(
        old.loc[list(members)], new, {site: record["layer"] for site, record in members.items()}
    )
    near = near_threshold_rows(long, new, gate_config["rules"], config["near_threshold_margin"])
    outputs = write_outputs(
        {"coverage_long": long, "flags_old_vs_new": comparison, "near_threshold": near}, config
    )
    frozen_path = REPO_ROOT / config["frozen_table"]
    frozen_table(old_full, new).to_csv(frozen_path, index_label="site")
    for path in [*outputs, frozen_path]:
        print(path)
    print(f"qc_level={config['qc_level']} sha256={file_sha256(REPO_ROOT / config['qc_config'])}")
