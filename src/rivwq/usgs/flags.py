from __future__ import annotations

import pandas as pd

DAYS_PER_YEAR = 365.25


def clipped_span(
    first: pd.Timestamp, last: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    if pd.isna(first) or pd.isna(last):
        return pd.NaT, pd.NaT, 0.0
    clipped_first = max(first, start)
    clipped_last = min(last, end)
    if clipped_last <= clipped_first:
        return pd.NaT, pd.NaT, 0.0
    return clipped_first, clipped_last, (clipped_last - clipped_first).days / DAYS_PER_YEAR


def add_fchl_columns(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    window_start, window_end = (pd.Timestamp(value) for value in config["fchl_window"])
    out_of_time_start = pd.Timestamp(config["fchl_out_of_time_start"])
    minimum_years = config["fchl_min_years_before_out_of_time"]
    firsts = pd.to_datetime(table["chlorophyll_fluorescence_first"], errors="coerce")
    lasts = pd.to_datetime(table["chlorophyll_fluorescence_last"], errors="coerce")
    result = table.copy()
    spans = [
        clipped_span(first, last, window_start, window_end)
        for first, last in zip(firsts, lasts, strict=True)
    ]
    result["fchl_first"] = [span[0] for span in spans]
    result["fchl_last"] = [span[1] for span in spans]
    result["fchl_years"] = [span[2] for span in spans]
    before = [
        clipped_span(first, last, window_start, out_of_time_start)[2]
        for first, last in zip(firsts, lasts, strict=True)
    ]
    result["fchl_before_2025"] = [years >= minimum_years for years in before]
    return result


def add_turbidity_only(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    missing_do = result["dissolved_oxygen_code"].isna()
    missing_fchl = result["chlorophyll_fluorescence_code"].isna()
    result["turbidity_only"] = missing_do | missing_fchl
    return result


def add_caveats(table: pd.DataFrame, caveat_sites: dict[str, str]) -> pd.DataFrame:
    result = table.copy()
    reasons = pd.Series(result.index.astype(str), index=result.index).map(caveat_sites)
    result["caveat"] = reasons.notna()
    result["caveat_reason"] = reasons.fillna("")
    return result


def add_all_flags(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    result = add_fchl_columns(table, config)
    result = add_turbidity_only(result)
    return add_caveats(result, config["caveat_sites"])
