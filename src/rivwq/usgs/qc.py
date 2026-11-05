from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "usgs_qc.yaml"
SERIES_KEY = ["site_no", "parameter_code", "sensor_id"]
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0
SECONDS_PER_MINUTE = 60.0
STATUS_PROVISIONAL = "Provisional"
STATUS_APPROVED = "Approved"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def observed(series: pd.DataFrame) -> pd.DataFrame:
    return series[series["value"].notna()].sort_values("datetime_utc")


def median_interval_seconds(times: pd.Series) -> float:
    unique = times.drop_duplicates().sort_values()
    if len(unique) < 2:
        return float("nan")
    return float(unique.diff().dropna().dt.total_seconds().median())


def duplicate_counts(series: pd.DataFrame) -> tuple[int, int]:
    repeated = series[series.duplicated("datetime_utc", keep=False)]
    extra_rows = int(series.duplicated("datetime_utc", keep="first").sum())
    conflicting = 0
    if not repeated.empty:
        spread = repeated.groupby("datetime_utc")["value"].nunique(dropna=False)
        conflicting = int((spread > 1).sum())
    return extra_rows, conflicting


def long_gaps(times: pd.Series, gap_days: float) -> pd.DataFrame:
    ordered = times.drop_duplicates().sort_values().reset_index(drop=True)
    steps = ordered.diff().dt.total_seconds() / SECONDS_PER_DAY
    mask = steps > gap_days
    return pd.DataFrame(
        {
            "gap_start_utc": ordered.shift(1)[mask].to_numpy(),
            "gap_end_utc": ordered[mask].to_numpy(),
            "gap_days": steps[mask].to_numpy(),
        }
    )


def flatline_runs(
    series: pd.DataFrame, interval_seconds: float, min_hours: float, break_intervals: float
) -> pd.DataFrame:
    data = observed(series).drop_duplicates("datetime_utc").reset_index(drop=True)
    columns = ["run_start_utc", "run_end_utc", "run_hours", "run_values", "value"]
    if data.empty:
        return pd.DataFrame(columns=columns)
    steps = data["datetime_utc"].diff().dt.total_seconds()
    changed = data["value"].ne(data["value"].shift())
    too_far = steps > break_intervals * interval_seconds
    data["run"] = (changed | too_far).cumsum()
    runs = data.groupby("run").agg(
        run_start_utc=("datetime_utc", "min"),
        run_end_utc=("datetime_utc", "max"),
        run_values=("value", "size"),
        value=("value", "first"),
    )
    runs["run_hours"] = (
        runs["run_end_utc"] - runs["run_start_utc"]
    ).dt.total_seconds() / SECONDS_PER_HOUR
    return runs[runs["run_hours"] > min_hours].reset_index(drop=True)[columns]


def ceiling_counts(values: pd.Series, ceiling: float | None, near_fraction: float) -> tuple:
    if ceiling is None:
        return None, None
    at_ceiling = int((values >= ceiling).sum())
    near_ceiling = int((values >= near_fraction * ceiling).sum())
    return at_ceiling, near_ceiling


def year_table(
    series: pd.DataFrame, interval_seconds: float, run_end: pd.Timestamp
) -> pd.DataFrame:
    data = observed(series)
    first, last = data["datetime_utc"].min(), data["datetime_utc"].max()
    rows = []
    for year in sorted(series["datetime_utc"].dt.year.unique()):
        start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        end = min(pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC"), run_end)
        rows_in_year = series[series["datetime_utc"].dt.year == year]
        values_in_year = data[data["datetime_utc"].dt.year == year]
        distinct = values_in_year["datetime_utc"].nunique()
        span_start, span_end = max(start, first), min(end, last)
        span_seconds = max((span_end - span_start).total_seconds(), 0.0)
        year_seconds = (end - start).total_seconds()
        status = rows_in_year["approval_status"]
        rows.append(
            {
                "year": year,
                "rows": len(rows_in_year),
                "values": len(values_in_year),
                "coverage_calendar": min(distinct / (year_seconds / interval_seconds), 1.0),
                "coverage_in_span": min(distinct / (span_seconds / interval_seconds + 1.0), 1.0)
                if pd.notna(first)
                else np.nan,
                "share_provisional": float((status == STATUS_PROVISIONAL).mean()),
                "share_approved": float((status == STATUS_APPROVED).mean()),
                "share_other_status": float(
                    (~status.isin([STATUS_PROVISIONAL, STATUS_APPROVED])).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def series_summary(series: pd.DataFrame, settings: dict) -> dict:
    data = observed(series)
    interval = median_interval_seconds(series["datetime_utc"])
    extra_rows, conflicting = duplicate_counts(series)
    gaps = long_gaps(data["datetime_utc"], settings["gap_days"])
    flats = flatline_runs(
        series, interval, settings["flatline_hours"], settings["flatline_break_intervals"]
    )
    code = series["parameter_code"].iloc[0]
    ceiling = settings["sensor_ceilings"].get(code)
    at_ceiling, near_ceiling = ceiling_counts(
        data["value"], ceiling, settings["ceiling_near_fraction"]
    )
    return {
        "site_no": series["site_no"].iloc[0],
        "parameter_code": code,
        "sensor_id": series["sensor_id"].iloc[0],
        "first_utc": data["datetime_utc"].min(),
        "last_utc": data["datetime_utc"].max(),
        "rows": len(series),
        "values": len(data),
        "null_values": int(series["value"].isna().sum()),
        "median_interval_minutes": interval / SECONDS_PER_MINUTE,
        "duplicate_timestamp_rows": extra_rows,
        "duplicate_timestamps_with_conflicting_values": conflicting,
        "negative_values": int((data["value"] < 0).sum()),
        "zero_values": int((data["value"] == 0).sum()),
        "flatline_runs_over_limit": len(flats),
        "longest_flatline_hours": float(flats["run_hours"].max()) if len(flats) else 0.0,
        "ceiling": ceiling,
        "values_at_ceiling": at_ceiling,
        "values_near_ceiling": near_ceiling,
        "qualified_values": int((series["qualifiers"] != "").sum()),
        "units_seen": "|".join(sorted(series["unit_of_measure"].unique())),
        "gaps_over_limit": len(gaps),
        "longest_gap_days": float(gaps["gap_days"].max()) if len(gaps) else 0.0,
    }


def qc_series(series: pd.DataFrame, settings: dict, run_end: pd.Timestamp) -> dict:
    key = {name: series[name].iloc[0] for name in SERIES_KEY}
    interval = median_interval_seconds(series["datetime_utc"])
    data = observed(series)
    gaps = long_gaps(data["datetime_utc"], settings["gap_days"])
    flats = flatline_runs(
        series, interval, settings["flatline_hours"], settings["flatline_break_intervals"]
    )
    years = year_table(series, interval, run_end)
    for table in (gaps, flats, years):
        for name, value in key.items():
            table.insert(list(key).index(name), name, value)
    return {
        "summary": series_summary(series, settings),
        "years": years,
        "gaps": gaps,
        "flatlines": flats,
    }


def qc_site(tidy: pd.DataFrame, settings: dict, run_end: pd.Timestamp) -> dict:
    parts = [
        qc_series(group, settings, run_end) for _, group in tidy.groupby(SERIES_KEY, sort=True)
    ]
    return {
        "summary": pd.DataFrame([part["summary"] for part in parts]),
        "years": pd.concat([part["years"] for part in parts], ignore_index=True),
        "gaps": pd.concat([part["gaps"] for part in parts], ignore_index=True),
        "flatlines": pd.concat([part["flatlines"] for part in parts], ignore_index=True),
    }


def read_site(run_dir: Path, site: str) -> pd.DataFrame:
    paths = sorted((run_dir / "tidy" / site).glob("*.parquet"))
    frames = [pd.read_parquet(path) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    settings = load_config(args.config)
    run_dir = REPO_ROOT / settings["raw_root"] / args.run_id
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    run_end = pd.Timestamp(state["started_utc"])
    output_dir = (args.output_dir or Path(settings["output_dir"])).expanduser()
    sites = sorted(path.name for path in (run_dir / "tidy").iterdir() if path.is_dir())
    results = []
    for site in sites:
        tidy = read_site(run_dir, site)
        if not tidy.empty:
            results.append(qc_site(tidy, settings, run_end))
    for name in ("summary", "years", "gaps", "flatlines"):
        combined = pd.concat([result[name] for result in results], ignore_index=True)
        suffix = "series" if name == "summary" else name
        combined.to_csv(output_dir / f"{settings['output_prefix']}_{suffix}.csv", index=False)
        print(f"{suffix}: {len(combined)} rows")
