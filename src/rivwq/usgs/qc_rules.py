from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "qc.yaml"
SERIES_KEY = ["site_no", "parameter_code", "sensor_id"]
RULES = [
    "value_null",
    "unreliable_qualifier",
    "physical_limit",
    "temperature_zero_run",
    "turbidity_ceiling",
    "flatline",
]
REPORT_ONLY_RULES = ["negative_discharge_stage"]
CARRIED_COLUMNS = [
    "site_no",
    "datetime_utc",
    "parameter_code",
    "sensor_id",
    "value",
    "qualifiers",
    "approval_status",
]
SECONDS_PER_HOUR = 3600.0


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def qualifier_pattern(codes: list[str]) -> re.Pattern:
    alternatives = "|".join(re.escape(code) for code in codes)
    return re.compile(rf"(?:^|\|)(?:{alternatives})(?:\||$)")


def unreliable_qualifier(tidy: pd.DataFrame, codes: list[str]) -> pd.Series:
    pattern = qualifier_pattern(codes)
    return tidy["qualifiers"].fillna("").str.contains(pattern)


def physical_limit(tidy: pd.DataFrame, limits: dict) -> pd.Series:
    flagged = pd.Series(False, index=tidy.index)
    for code, bounds in limits.items():
        in_code = tidy["parameter_code"] == code
        if "min" in bounds:
            flagged |= in_code & (tidy["value"] < bounds["min"])
        if "max" in bounds:
            flagged |= in_code & (tidy["value"] > bounds["max"])
    return flagged


def negative_discharge_stage(tidy: pd.DataFrame, codes: list[str]) -> pd.Series:
    return tidy["parameter_code"].isin(codes) & (tidy["value"] < 0)


def median_interval_seconds(times: pd.Series) -> float:
    unique = times.drop_duplicates().sort_values()
    if len(unique) < 2:
        return float("nan")
    return float(unique.diff().dropna().dt.total_seconds().median())


def run_statistics(series: pd.DataFrame, break_intervals: float) -> pd.DataFrame:
    observed = series[series["value"].notna()].sort_values("datetime_utc")
    columns = ["run_values", "run_span_hours"]
    if len(observed) < 2:
        return pd.DataFrame({column: 0.0 for column in columns}, index=observed.index)
    interval = median_interval_seconds(observed["datetime_utc"])
    steps = observed["datetime_utc"].diff().dt.total_seconds()
    starts = observed["value"].ne(observed["value"].shift()) | (steps > break_intervals * interval)
    run = starts.cumsum()
    grouped = observed.groupby(run)
    first = grouped["datetime_utc"].transform("min")
    last = grouped["datetime_utc"].transform("max")
    return pd.DataFrame(
        {
            "run_values": grouped["value"].transform("size").astype(float),
            "run_span_hours": ((last - first).dt.total_seconds() + interval) / SECONDS_PER_HOUR,
        },
        index=observed.index,
    )


def run_flags(tidy: pd.DataFrame, codes: list[str], break_intervals: float) -> pd.DataFrame:
    selected = tidy[tidy["parameter_code"].isin(codes)]
    parts = [run_statistics(group, break_intervals) for _, group in selected.groupby(SERIES_KEY)]
    columns = ["run_values", "run_span_hours"]
    if not parts:
        return pd.DataFrame(0.0, index=tidy.index, columns=columns)
    return pd.concat(parts).reindex(tidy.index).fillna(0.0)


def temperature_zero_run(tidy: pd.DataFrame, settings: dict, break_intervals: float) -> pd.Series:
    code = settings["parameter_code"]
    runs = run_flags(tidy, [code], break_intervals)
    at_value = (tidy["parameter_code"] == code) & (tidy["value"] == settings["value"])
    return at_value & (runs["run_values"] >= settings["min_run_values"])


def turbidity_ceiling(tidy: pd.DataFrame, settings: dict) -> pd.Series:
    ceilings = tidy["sensor_id"].map(settings["ceilings"])
    in_code = tidy["parameter_code"] == settings["parameter_code"]
    return in_code & (tidy["value"] >= ceilings)


def flatline(tidy: pd.DataFrame, limits: dict, break_intervals: float) -> pd.Series:
    runs = run_flags(tidy, list(limits), break_intervals)
    allowed = tidy["parameter_code"].map(limits)
    return runs["run_span_hours"] > allowed


def compute_rules(tidy: pd.DataFrame, config: dict) -> pd.DataFrame:
    break_intervals = config["run_break_intervals"]
    flags = tidy[CARRIED_COLUMNS].copy()
    flags["value_null"] = tidy["value"].isna()
    flags["unreliable_qualifier"] = unreliable_qualifier(tidy, config["unreliable_qualifiers"])
    flags["physical_limit"] = physical_limit(tidy, config["physical_limits"])
    flags["temperature_zero_run"] = temperature_zero_run(
        tidy, config["temperature_zero_run"], break_intervals
    )
    flags["turbidity_ceiling"] = turbidity_ceiling(tidy, config["turbidity_ceiling"])
    flags["flatline"] = flatline(tidy, config["flatline"]["max_run_hours"], break_intervals)
    flags["negative_discharge_stage"] = negative_discharge_stage(
        tidy, config["report_negative_codes"]
    )
    return flags


def restrict_to_level(flags: pd.DataFrame, config: dict, level: str) -> pd.DataFrame:
    active = config["levels"][level]
    result = flags.copy()
    for rule in RULES:
        if rule not in active:
            result[rule] = False
    result["exclude"] = result[RULES].any(axis=1)
    return result


def apply_level(tidy: pd.DataFrame, config: dict, level: str) -> pd.DataFrame:
    return restrict_to_level(compute_rules(tidy, config), config, level)


def count_flags(flags: pd.DataFrame, level: str) -> pd.DataFrame:
    rules = RULES + REPORT_ONLY_RULES + ["exclude"]
    grouped = flags.groupby(["site_no", "parameter_code"])
    totals = grouped.size().rename("rows")
    counts = grouped[rules].sum().astype(int)
    table = counts.join(totals).reset_index()
    long = table.melt(
        id_vars=["site_no", "parameter_code", "rows"], var_name="rule", value_name="flagged"
    )
    long.insert(0, "level", level)
    return long[["level", "site_no", "parameter_code", "rule", "rows", "flagged"]]


def read_site(run_dir: Path, site: str) -> pd.DataFrame:
    paths = sorted((run_dir / "tidy" / site).glob("*.parquet"))
    frames = [pd.read_parquet(path) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rule_counts(run_dir: Path, config: dict) -> pd.DataFrame:
    tables = []
    for site_dir in sorted(path for path in (run_dir / "tidy").iterdir() if path.is_dir()):
        tidy = read_site(run_dir, site_dir.name)
        if tidy.empty:
            continue
        flags = compute_rules(tidy, config)
        for level in config["levels"]:
            tables.append(count_flags(restrict_to_level(flags, config, level), level))
    return pd.concat(tables, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    counts = rule_counts(REPO_ROOT / config["raw_root"] / args.run_id, config)
    target = Path(config["output_dir"]).expanduser() / config["counts_file"]
    counts.to_csv(target, index=False)
    print(f"{target}: {len(counts)} rows")
