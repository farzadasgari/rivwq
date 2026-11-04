from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from rivwq.usgs.scout import value_columns

DAILY_COLUMNS = ["date", "mean", "maximum"]
STATISTIC_NAMES = [
    "turbidity_median",
    "turbidity_p95",
    "turbidity_max",
    "turbidity_share_above_200",
]


def best_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    if not candidates:
        return None
    counts = {name: int(frame[name].notna().sum()) for name in candidates}
    return max(counts, key=lambda name: (counts[name], name))


def condense_daily(frame: pd.DataFrame | None, code: str) -> pd.DataFrame:
    empty = pd.DataFrame(columns=DAILY_COLUMNS)
    if frame is None or frame.empty:
        return empty
    columns = value_columns(frame, code)
    means = [name for name in columns if name.endswith("_Mean")]
    mean_name = best_column(frame, means or columns)
    if mean_name is None:
        return empty
    maximum_name = mean_name.removesuffix("_Mean") + "_Maximum"
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    if maximum_name in frame.columns:
        maximum = pd.to_numeric(frame[maximum_name], errors="coerce").to_numpy()
    else:
        maximum = np.full(len(frame), np.nan)
    result = pd.DataFrame(
        {
            "date": index.normalize(),
            "mean": pd.to_numeric(frame[mean_name], errors="coerce").to_numpy(),
            "maximum": maximum,
        }
    )
    return result.dropna(subset=["mean"]).drop_duplicates("date").reset_index(drop=True)


def window_days(year: int) -> int:
    return (pd.Timestamp(year + 1, 1, 1) - pd.Timestamp(year, 1, 1)).days


def unique_days(present_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(present_days).normalize().unique().sort_values()


def yearly_coverage(present_days: pd.DatetimeIndex, years: list[int]) -> dict[int, float]:
    days = unique_days(present_days)
    return {year: float((days.year == year).sum() / window_days(year)) for year in years}


def longest_gap_days(present_days: pd.DatetimeIndex, start: str, end: str) -> int:
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    days = unique_days(present_days)
    days = days[(days >= first) & (days <= last)]
    if len(days) == 0:
        return (last - first).days + 1
    edges = pd.DatetimeIndex([first - pd.Timedelta(days=1), *days, last + pd.Timedelta(days=1)])
    gaps = np.diff(edges.to_numpy()).astype("timedelta64[D]").astype(int) - 1
    return int(gaps.max())


def longest_interior_gap_days(present_days: pd.DatetimeIndex, start: str, end: str) -> int:
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    days = unique_days(present_days)
    days = days[(days >= first) & (days <= last)]
    if len(days) < 2:
        return 0
    gaps = np.diff(days.to_numpy()).astype("timedelta64[D]").astype(int) - 1
    return int(gaps.max())


def turbidity_statistics(daily: pd.DataFrame, high_fnu: float) -> dict[str, float]:
    if daily.empty:
        return dict.fromkeys(STATISTIC_NAMES, np.nan)
    means = daily["mean"].astype(float)
    peaks = daily["maximum"].astype(float).where(daily["maximum"].notna(), means)
    return {
        "turbidity_median": float(means.median()),
        "turbidity_p95": float(means.quantile(0.95)),
        "turbidity_max": float(peaks.max()),
        "turbidity_share_above_200": float((means > high_fnu).mean()),
    }


def fallback_windows(
    years: list[int], starts: list[str], length_days: int
) -> list[tuple[str, str]]:
    windows = []
    for year in years:
        for month_day in starts:
            begin = pd.Timestamp(f"{year}-{month_day}")
            end = begin + pd.Timedelta(days=length_days - 1)
            windows.append((begin.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    return windows


def sampled_coverage(
    present_days: pd.DatetimeIndex, windows: list[tuple[str, str]], years: list[int]
) -> dict[int, float]:
    days = unique_days(present_days)
    result = {}
    for year in years:
        year_windows = [pair for pair in windows if pair[0].startswith(str(year))]
        total = 0
        hit = 0
        for begin, end in year_windows:
            total += (pd.Timestamp(end) - pd.Timestamp(begin)).days + 1
            hit += int(((days >= pd.Timestamp(begin)) & (days <= pd.Timestamp(end))).sum())
        result[year] = hit / total if total else np.nan
    return result


def with_retries(call, attempts: int, pause: float):
    for attempt in range(attempts):
        try:
            return call()
        except Exception:
            if attempt == attempts - 1:
                return None
            time.sleep(pause * 2**attempt)
    return None


def fetch_daily(site: str, code: str, start: str, end: str) -> pd.DataFrame:
    from dataretrieval import nwis

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return nwis.get_dv(sites=site, parameterCd=code, start=start, end=end)[0]


def fetch_instantaneous(site: str, code: str, start: str, end: str) -> pd.DataFrame:
    from dataretrieval import nwis

    return nwis.get_iv(sites=site, parameterCd=code, start=start, end=end)[0]


def daily_from_instantaneous(frame: pd.DataFrame | None, code: str) -> pd.DataFrame:
    empty = pd.DataFrame(columns=DAILY_COLUMNS)
    if frame is None or frame.empty:
        return empty
    columns = value_columns(frame, code)
    name = best_column(frame, columns)
    if name is None:
        return empty
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    values = pd.Series(pd.to_numeric(frame[name], errors="coerce").to_numpy(), index=index)
    values = values.dropna()
    if values.empty:
        return empty
    grouped = values.groupby(values.index.normalize())
    result = pd.DataFrame({"mean": grouped.mean(), "maximum": grouped.max()})
    return result.rename_axis("date").reset_index()


def study_years(settings: dict) -> list[int]:
    return list(range(int(settings["start"][:4]), int(settings["end"][:4]) + 1))


def daily_path(cache: Path, site: str, code: str) -> Path:
    return cache / "dv" / f"{site}_{code}.csv"


def sample_path(cache: Path, site: str, code: str) -> Path:
    return cache / "iv_sample" / f"{site}_{code}.csv"


def all_windows(settings: dict) -> list[tuple[str, str]]:
    return fallback_windows(
        study_years(settings), settings["fallback_window_starts"], settings["fallback_window_days"]
    )


def tail_windows(settings: dict, last_date: pd.Timestamp) -> list[tuple[str, str]]:
    margin = pd.Timedelta(days=settings["tail_margin_days"])
    if last_date >= pd.Timestamp(settings["end"]) - margin:
        return []
    return [pair for pair in all_windows(settings) if pd.Timestamp(pair[0]) > last_date]


def tail_path(cache: Path, site: str, code: str) -> Path:
    return cache / "iv_tail" / f"{site}_{code}.csv"


def fetch_sampled_days(
    site: str, code: str, settings: dict, windows: list[tuple[str, str]]
) -> pd.DataFrame | None:
    attempts, wait = settings["retry_attempts"], settings["retry_pause_seconds"]
    frames: list[pd.DataFrame] = []
    for begin, end in windows:
        sample = with_retries(
            lambda begin=begin, end=end: fetch_instantaneous(site, code, begin, end),
            attempts,
            wait,
        )
        time.sleep(settings["request_pause_seconds"])
        if sample is None:
            return None
        frames.append(daily_from_instantaneous(sample, code))
    return pd.concat(frames, ignore_index=True)


def fetch_series(site: str, code: str, settings: dict, cache: Path) -> str:
    daily_file = daily_path(cache, site, code)
    sample_file = sample_path(cache, site, code)
    if daily_file.exists() or sample_file.exists():
        return "cached"
    attempts, wait = settings["retry_attempts"], settings["retry_pause_seconds"]
    frame = with_retries(
        lambda: fetch_daily(site, code, settings["start"], settings["end"]), attempts, wait
    )
    time.sleep(settings["request_pause_seconds"])
    if frame is None:
        return "failed"
    daily = condense_daily(frame, code)
    if not daily.empty:
        daily_file.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(daily_file, index=False)
        return "daily"
    sampled = fetch_sampled_days(site, code, settings, all_windows(settings))
    if sampled is None:
        return "failed"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(sample_file, index=False)
    return "iv_sample"


def fetch_tail(site: str, code: str, settings: dict, cache: Path) -> str:
    daily_file = daily_path(cache, site, code)
    tail_file = tail_path(cache, site, code)
    if not daily_file.exists() or tail_file.exists():
        return "none"
    last_date = pd.read_csv(daily_file, parse_dates=["date"])["date"].max()
    windows = tail_windows(settings, last_date)
    if not windows:
        return "none"
    sampled = fetch_sampled_days(site, code, settings, windows)
    if sampled is None:
        return "failed"
    tail_file.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(tail_file, index=False)
    return "tail"


def fetch_job(site: str, code: str, settings: dict, cache: Path) -> str:
    outcome = fetch_series(site, code, settings, cache)
    if outcome == "failed":
        return outcome
    tail = fetch_tail(site, code, settings, cache)
    return "failed" if tail == "failed" else outcome


def fetch_all(
    jobs: list[tuple[str, str]], settings: dict, cache: Path
) -> dict[tuple[str, str], str]:
    outcomes = {}
    with ThreadPoolExecutor(max_workers=settings["workers"]) as pool:
        futures = {job: pool.submit(fetch_job, job[0], job[1], settings, cache) for job in jobs}
        for job, future in futures.items():
            outcomes[job] = future.result()
            print(f"{outcomes[job]} {job[0]} {job[1]}")
    return outcomes


def load_series(site: str, code: str, cache: Path) -> tuple[str, pd.DataFrame]:
    daily_file = daily_path(cache, site, code)
    if daily_file.exists():
        return "daily", pd.read_csv(daily_file, parse_dates=["date"])
    sample_file = sample_path(cache, site, code)
    if sample_file.exists():
        return "iv_sample", pd.read_csv(sample_file, parse_dates=["date"])
    return "missing", pd.DataFrame(columns=DAILY_COLUMNS)


def load_tail(site: str, code: str, cache: Path) -> pd.DataFrame:
    path = tail_path(cache, site, code)
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return pd.DataFrame(columns=DAILY_COLUMNS)


def turbidity_row(source: str, content: pd.DataFrame, tail: pd.DataFrame, settings: dict) -> dict:
    no_gaps = {"turbidity_gap_days": np.nan, "turbidity_interior_gap_days": np.nan}
    if source not in ("daily", "iv_sample"):
        return {**dict.fromkeys(STATISTIC_NAMES, np.nan), **no_gaps}
    combined = pd.concat([content, tail], ignore_index=True) if len(tail) else content
    row = turbidity_statistics(combined, settings["turbidity_high_fnu"])
    if source == "iv_sample":
        return {**row, **no_gaps}
    present = pd.DatetimeIndex(content["date"])
    last_covered = settings["end"]
    if len(tail) or tail_windows(settings, present.max()):
        last_covered = present.max().strftime("%Y-%m-%d")
    return {
        **row,
        "turbidity_gap_days": longest_gap_days(present, settings["start"], last_covered),
        "turbidity_interior_gap_days": longest_interior_gap_days(
            present, settings["start"], settings["end"]
        ),
        "turbidity_daily_last": present.max(),
    }


def mean_or_nan(values: list[float]) -> float:
    present = [value for value in values if not np.isnan(value)]
    return float(np.mean(present)) if present else np.nan


def tail_adjusted_coverage(
    content: pd.DataFrame, tail: pd.DataFrame, settings: dict
) -> dict[int, float]:
    years = study_years(settings)
    present = unique_days(pd.DatetimeIndex(content["date"]))
    last_date = present.max()
    windows = tail_windows(settings, last_date)
    daily = yearly_coverage(present, years)
    tail_days = pd.DatetimeIndex(tail["date"]) if len(tail) else pd.DatetimeIndex([])
    result = {}
    for year in years:
        year_windows = [pair for pair in windows if pair[0].startswith(str(year))]
        if not year_windows:
            result[year] = daily[year]
            continue
        share = sampled_coverage(tail_days, year_windows, [year])[year]
        year_end = pd.Timestamp(year, 12, 31)
        year_start = pd.Timestamp(year, 1, 1)
        first_uncovered = max(last_date + pd.Timedelta(days=1), year_start)
        remaining = (year_end - first_uncovered).days + 1
        dv_days = daily[year] * window_days(year)
        result[year] = (dv_days + share * remaining) / window_days(year)
    return result


def estimated_years(source: str, content: pd.DataFrame, settings: dict) -> list[int]:
    years = study_years(settings)
    if source == "iv_sample":
        return years
    if source != "daily" or content.empty:
        return []
    windows = tail_windows(settings, pd.DatetimeIndex(content["date"]).max())
    return [year for year in years if any(pair[0].startswith(str(year)) for pair in windows)]


def variable_coverage(source: str, content: pd.DataFrame, tail: pd.DataFrame, settings: dict):
    years = study_years(settings)
    if source == "daily":
        return tail_adjusted_coverage(content, tail, settings)
    if source == "iv_sample":
        return sampled_coverage(pd.DatetimeIndex(content["date"]), all_windows(settings), years)
    return dict.fromkeys(years, np.nan)


def site_coverage_row(site: str, codes: dict[str, str | None], settings: dict, cache: Path) -> dict:
    row: dict = {"site_no": site}
    for variable, code in codes.items():
        tail = pd.DataFrame(columns=DAILY_COLUMNS)
        if code is None:
            source, content = "no_series", pd.DataFrame(columns=DAILY_COLUMNS)
            coverage = dict.fromkeys(study_years(settings), 0.0)
        else:
            source, content = load_series(site, code, cache)
            tail = load_tail(site, code, cache)
            coverage = variable_coverage(source, content, tail, settings)
        row[f"{variable}_source"] = "daily+iv_tail" if source == "daily" and len(tail) else source
        for year, share in coverage.items():
            row[f"{variable}_cov_{year}"] = share
        row[f"{variable}_cov_mean"] = mean_or_nan(list(coverage.values()))
        row[f"{variable}_estimated_years"] = ";".join(
            str(year) for year in estimated_years(source, content, settings)
        )
        if variable == "turbidity":
            row.update(turbidity_row(source, content, tail, settings))
    return row
