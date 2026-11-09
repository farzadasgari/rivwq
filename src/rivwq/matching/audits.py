from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rivwq.analysis import site_rules_iv
from rivwq.matching import readings, table
from rivwq.usgs import qc_rules

NONE_TOKEN = "(none)"
QUALIFIER_SEPARATOR = "|"
MIN_BLOCKS = 1


def native_summary(natives: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in natives.groupby(["site_no", "parameter_code", "sensor_id"], sort=True):
        valid = group.dropna(subset=["native_minutes"])
        weights = valid.groupby("native_minutes")["values"].sum()
        rows.append(
            {
                "site_no": key[0],
                "parameter_code": key[1],
                "sensor_id": key[2],
                "first_year": int(group["year"].min()),
                "last_year": int(group["year"].max()),
                "native_minutes_modal": float(weights.idxmax()) if len(weights) else np.nan,
                "native_minutes_seen": "|".join(f"{v:g}" for v in sorted(weights.index)),
                "years_with_interval": len(valid),
                "values": int(group["values"].sum()),
            }
        )
    return pd.DataFrame(rows)


def sensor_spans(kept: pd.DataFrame) -> pd.DataFrame:
    grouped = kept.groupby(["site_no", "parameter_code", "sensor_id"], sort=True)
    spans = grouped["datetime_utc"].agg(first_utc="min", last_utc="max", values="size")
    return spans.reset_index()


def token_rows(tidy: pd.DataFrame, flags: pd.DataFrame, unreliable: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "site_no": tidy["site_no"].to_numpy(),
            "year": tidy["datetime_utc"].dt.year.to_numpy(),
            "parameter_code": tidy["parameter_code"].to_numpy(),
            "approval_status": tidy["approval_status"].to_numpy(),
            "kept": (~flags["exclude"]).to_numpy(),
            "token": tidy["qualifiers"].fillna("").str.split(QUALIFIER_SEPARATOR).to_numpy(),
        }
    )
    frame = frame.explode("token")
    frame["token"] = frame["token"].replace("", NONE_TOKEN).fillna(NONE_TOKEN)
    keys = ["site_no", "year", "parameter_code", "approval_status", "token"]
    counts = frame.groupby(keys, sort=True).agg(rows=("kept", "size"), rows_kept=("kept", "sum"))
    counts = counts.reset_index()
    counts["rows_kept"] = counts["rows_kept"].astype(int)
    counts["in_unreliable_list"] = counts["token"].isin(unreliable)
    return counts


def qualifier_summary(counts: pd.DataFrame) -> pd.DataFrame:
    grouped = counts.groupby(["token", "parameter_code", "approval_status"], sort=True)
    summary = grouped.agg(rows=("rows", "sum"), rows_kept=("rows_kept", "sum")).reset_index()
    site_years = (
        counts.drop_duplicates(["token", "site_no", "year"])
        .groupby("token")
        .size()
        .rename("site_years")
    )
    summary = summary.join(site_years, on="token")
    summary["in_unreliable_list"] = summary["token"].isin(
        counts.loc[counts["in_unreliable_list"], "token"].unique()
    )
    return summary


def named_qualifier_table(counts: pd.DataFrame, named: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for label, tokens in named.items():
        present = counts[counts["token"].isin(tokens)]
        rows.append(
            {
                "label": label,
                "tokens_searched": QUALIFIER_SEPARATOR.join(tokens),
                "tokens_found": QUALIFIER_SEPARATOR.join(sorted(present["token"].unique())),
                "rows": int(present["rows"].sum()),
                "rows_kept_at_standard": int(present["rows_kept"].sum()),
                "site_years": int(present.drop_duplicates(["site_no", "year"]).shape[0]),
                "in_unreliable_list": bool(present["in_unreliable_list"].any()),
            }
        )
    return pd.DataFrame(rows)


def provisional_counts(tidy: pd.DataFrame, flags: pd.DataFrame, provisional: str) -> pd.DataFrame:
    kept = tidy[~flags["exclude"].to_numpy()]
    frame = pd.DataFrame(
        {
            "site_no": kept["site_no"].to_numpy(),
            "parameter_code": kept["parameter_code"].to_numpy(),
            "year": kept["datetime_utc"].dt.year.to_numpy(),
            "provisional": (kept["approval_status"] == provisional).to_numpy(),
        }
    )
    grouped = frame.groupby(["site_no", "parameter_code", "year"], sort=True)["provisional"]
    return grouped.agg(rows="size", provisional_rows="sum").reset_index()


def provisional_by_year(counts: pd.DataFrame) -> pd.DataFrame:
    pooled = counts.groupby(["parameter_code", "year"], sort=True)[
        ["rows", "provisional_rows"]
    ].sum()
    pooled["provisional_share"] = pooled["provisional_rows"] / pooled["rows"]
    return pooled.reset_index()


def provisional_in_year(counts: pd.DataFrame, year: int) -> pd.DataFrame:
    selected = counts[counts["year"] == year].copy()
    selected["provisional_share"] = selected["provisional_rows"] / selected["rows"]
    return selected.reset_index(drop=True)


def do_coverage_for_year(
    flags: pd.DataFrame, tidy: pd.DataFrame, code: str, year: int, cap: float
) -> pd.DataFrame:
    result = site_rules_iv.parameter_coverage(flags, tidy, code, [year], cap)
    result.insert(0, "site_no", tidy["site_no"].iloc[0])
    return result


def daily_mean(times: pd.Series, values: pd.Series) -> pd.Series:
    days = times.dt.tz_convert("UTC").dt.floor("D")
    return values.groupby(days.to_numpy()).mean()


def design_matrix(
    turbidity: pd.Series, discharge: pd.Series, before: tuple, after: tuple
) -> pd.DataFrame:
    joined = pd.concat({"turbidity": turbidity, "discharge": discharge}, axis=1, sort=True).dropna()
    joined = joined[(joined["turbidity"] > 0) & (joined["discharge"] > 0)]
    first = joined[(joined.index >= before[0]) & (joined.index < before[1])]
    second = joined[(joined.index >= after[0]) & (joined.index < after[1])]
    design = pd.concat([first.assign(new=0.0), second.assign(new=1.0)])
    design["ln_turbidity"] = np.log(design["turbidity"])
    design["ln_discharge"] = np.log(design["discharge"])
    return design


def step_coefficient(design: pd.DataFrame) -> float:
    exog = np.column_stack(
        [np.ones(len(design)), design["ln_discharge"].to_numpy(), design["new"].to_numpy()]
    )
    solution, *_ = np.linalg.lstsq(exog, design["ln_turbidity"].to_numpy(), rcond=None)
    return float(solution[2])


def resample_blocks(rows: pd.DataFrame, block: int, rng: np.random.Generator) -> pd.DataFrame:
    count = len(rows)
    blocks = max(int(np.ceil(count / block)), MIN_BLOCKS)
    starts = rng.integers(0, max(count - block + 1, 1), size=blocks)
    positions = np.concatenate([np.arange(start, min(start + block, count)) for start in starts])
    return rows.iloc[positions[:count]]


def bootstrap_interval(design: pd.DataFrame, settings: dict) -> tuple[float, float]:
    rng = np.random.default_rng(settings["bootstrap_seed"])
    sides = [design[design["new"] == 0.0], design[design["new"] == 1.0]]
    draws = []
    for _ in range(settings["bootstrap_resamples"]):
        sample = pd.concat(
            [resample_blocks(side, settings["bootstrap_block_days"], rng) for side in sides]
        )
        draws.append(step_coefficient(sample))
    tail = (1.0 - settings["interval_level"]) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return float(low), float(high)


def step_row(design: pd.DataFrame, settings: dict, label: str, with_interval: bool) -> dict:
    sides = design["new"].value_counts()
    days_before = int(sides.get(0.0, 0))
    days_after = int(sides.get(1.0, 0))
    row = {
        "label": label,
        "days_before": days_before,
        "days_after": days_after,
        "median_turbidity_before": float(design.loc[design["new"] == 0.0, "turbidity"].median()),
        "median_turbidity_after": float(design.loc[design["new"] == 1.0, "turbidity"].median()),
        "median_discharge_before": float(design.loc[design["new"] == 0.0, "discharge"].median()),
        "median_discharge_after": float(design.loc[design["new"] == 1.0, "discharge"].median()),
        "coefficient": np.nan,
        "ratio": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
    }
    enough = min(days_before, days_after) >= settings["min_days_per_side"]
    if not enough:
        return row
    row["coefficient"] = step_coefficient(design)
    row["ratio"] = float(np.exp(row["coefficient"]))
    if with_interval:
        row["ci_low"], row["ci_high"] = bootstrap_interval(design, settings)
    return row


def changeover_windows(spans: pd.DataFrame, days: int) -> dict:
    ordered = spans.sort_values("first_utc").reset_index(drop=True)
    old, new = ordered.iloc[0], ordered.iloc[-1]
    width = pd.Timedelta(days=days)
    old_end = old["last_utc"].floor("D")
    new_start = new["first_utc"].floor("D")
    return {
        "old_sensor": old["sensor_id"],
        "new_sensor": new["sensor_id"],
        "old_last_utc": old["last_utc"],
        "new_first_utc": new["first_utc"],
        "overlap": bool(old["last_utc"] >= new["first_utc"]),
        "before": (old_end - width, old_end + pd.Timedelta(days=1)),
        "after": (new_start, new_start + width),
    }


def shift_year(window: tuple, years: int) -> tuple:
    return tuple(stamp + pd.DateOffset(years=years) for stamp in window)


def changeover_test(
    kept_turbidity: pd.DataFrame, kept_discharge: pd.DataFrame, settings: dict
) -> pd.DataFrame:
    spans = sensor_spans(kept_turbidity)
    if len(spans) < 2:
        return pd.DataFrame()
    windows = changeover_windows(spans, settings["changeover_window_days"])
    turbidity = daily_mean(kept_turbidity["datetime_utc"], kept_turbidity["value"])
    discharge = daily_mean(kept_discharge["datetime_utc"], kept_discharge["value"])
    real = design_matrix(turbidity, discharge, windows["before"], windows["after"])
    rows = [step_row(real, settings, "changeover", True)]
    rows[0].update(
        {
            "old_sensor": windows["old_sensor"],
            "new_sensor": windows["new_sensor"],
            "old_last_utc": windows["old_last_utc"],
            "new_first_utc": windows["new_first_utc"],
            "sensors_overlap": windows["overlap"],
        }
    )
    changeover_year = windows["old_last_utc"].year
    for year in range(settings["placebo_first_year"], settings["placebo_last_year"] + 1):
        if year == changeover_year:
            continue
        shift = year - changeover_year
        before, after = shift_year(windows["before"], shift), shift_year(windows["after"], shift)
        window_rows = kept_turbidity[
            (kept_turbidity["datetime_utc"] >= before[0])
            & (kept_turbidity["datetime_utc"] < after[1])
        ]
        if window_rows["sensor_id"].nunique() != 1:
            continue
        design = design_matrix(turbidity, discharge, before, after)
        if design.empty:
            continue
        rows.append(step_row(design, settings, f"placebo_{year}", False))
    result = pd.DataFrame(rows)
    return result


def step_reading(result: pd.DataFrame) -> str:
    real = result[result["label"] == "changeover"].iloc[0]
    placebo = result[result["label"].str.startswith("placebo")]["coefficient"].abs().dropna()
    if np.isnan(real["coefficient"]):
        return "not testable: too few days"
    excludes_zero = real["ci_low"] > 0 or real["ci_high"] < 0
    largest = float(placebo.max()) if len(placebo) else np.nan
    beyond = bool(len(placebo)) and abs(real["coefficient"]) > largest
    if excludes_zero and beyond:
        return "step supported"
    return "not distinguishable from zero"


def kept_series(tidy: pd.DataFrame, flags: pd.DataFrame, code: str) -> pd.DataFrame:
    selected = (tidy["parameter_code"] == code).to_numpy() & ~flags["exclude"].to_numpy()
    return tidy[selected]


def audit_site(tidy: pd.DataFrame, config: dict, qc_config: dict) -> dict[str, pd.DataFrame]:
    flags = qc_rules.apply_level(tidy, qc_config, config["qc_primary"])
    audit = config["audit"]
    kept = tidy[~flags["exclude"].to_numpy()]
    parts = {
        "natives": readings.native_interval_table(tidy),
        "spans": sensor_spans(kept),
        "tokens": token_rows(tidy, flags, qc_config["unreliable_qualifiers"]),
        "provisional": provisional_counts(tidy, flags, config["provisional_status"]),
        "do_coverage": do_coverage_for_year(
            flags, tidy, config["parameters"]["dissolved_oxygen"], audit["report_year"], 1.0
        ),
    }
    if tidy["site_no"].iloc[0] == audit["changeover_site"]:
        parts["changeover"] = changeover_test(
            kept_series(tidy, flags, audit["changeover_parameter"]),
            kept_series(tidy, flags, audit["discharge_parameter"]),
            audit,
        )
    return parts


def waived_sites(config: dict, repo_root: Path, frozen: list[str]) -> list[str]:
    frame = pd.read_csv(repo_root / config["audit"]["waiver_table"], dtype={"site": str})
    flagged = frame[frame["legacy_do_waiver"].astype(bool)]["site"]
    return sorted(set(flagged) & set(frozen))


def run(config: dict, repo_root: Path, only_sites: list[str] | None) -> dict[str, pd.DataFrame]:
    qc_config = qc_rules.load_config()
    sites = table.site_layers(table.read_yaml(repo_root / config["sites_file"]))
    usgs_dir = repo_root / config["usgs_run_dir"]
    selected = only_sites or sorted(sites)
    collected: dict[str, list[pd.DataFrame]] = {}
    for site_no in selected:
        tidy = qc_rules.read_site(usgs_dir, site_no)
        if tidy.empty:
            continue
        for name, part in audit_site(tidy, config, qc_config).items():
            collected.setdefault(name, []).append(part)
    combined = {name: pd.concat(parts, ignore_index=True) for name, parts in collected.items()}
    return finalise(combined, sites, waived_sites(config, repo_root, list(sites)), config)


def finalise(
    combined: dict[str, pd.DataFrame], sites: dict[str, dict], waived: list[str], config: dict
) -> dict[str, pd.DataFrame]:
    audit = config["audit"]
    layers = pd.DataFrame(
        [
            {"site_no": site, "site_layer": item["layer"], "cv_group": item["cv_group"]}
            for site, item in sites.items()
        ]
    )
    natives = native_summary(combined["natives"])
    coverage = pd.DataFrame({"site_no": waived}).merge(
        combined["do_coverage"], on="site_no", how="left"
    )
    coverage["has_do_series"] = coverage["sensors"].notna()
    result = {
        "native_intervals": natives.merge(layers, on="site_no", how="left"),
        "sensor_spans": combined["spans"],
        "do_waived_coverage": coverage.merge(layers, on="site_no", how="left"),
        "qualifiers": qualifier_summary(combined["tokens"]),
        "qualifiers_named": named_qualifier_table(combined["tokens"], audit["named_qualifiers"]),
        "provisional_by_year": provisional_by_year(combined["provisional"]),
        "provisional_report_year": provisional_in_year(
            combined["provisional"], audit["report_year"]
        ),
    }
    if "changeover" in combined:
        result["changeover"] = combined["changeover"]
    return result


def write(result: dict[str, pd.DataFrame], report_dir: Path) -> list[Path]:
    paths = []
    for name, frame in result.items():
        path = report_dir / f"06_{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths
