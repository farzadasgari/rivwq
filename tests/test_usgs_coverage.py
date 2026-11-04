import numpy as np
import pandas as pd
import pytest

from rivwq.usgs import coverage


def daily_frame(days: list[str], means: list[float], maxima: list[float] | None = None):
    index = pd.DatetimeIndex(days, tz="UTC", name="datetime")
    data = {"63680_Mean": means, "63680_Mean_cd": ["A"] * len(days)}
    if maxima is not None:
        data["63680_Maximum"] = maxima
    return pd.DataFrame(data, index=index)


def test_condense_daily_keeps_mean_and_maximum_without_codes_columns():
    frame = daily_frame(["2024-01-01", "2024-01-02"], [5.0, np.nan], [9.0, np.nan])
    result = coverage.condense_daily(frame, "63680")
    assert list(result.columns) == coverage.DAILY_COLUMNS
    assert list(result["date"]) == [pd.Timestamp("2024-01-01")]
    assert list(result["maximum"]) == [9.0]


def test_condense_daily_empty_input_returns_empty_frame():
    assert coverage.condense_daily(pd.DataFrame(), "63680").empty
    assert coverage.condense_daily(None, "63680").empty


def test_yearly_coverage_counts_days_over_year_length():
    days = pd.date_range("2024-01-01", periods=183)
    result = coverage.yearly_coverage(days, [2023, 2024])
    assert result[2023] == 0.0
    assert result[2024] == pytest.approx(183 / 366)


def test_duplicate_days_are_counted_once():
    days = pd.DatetimeIndex(["2025-03-01 00:00", "2025-03-01 12:00"])
    assert coverage.yearly_coverage(days, [2025])[2025] == pytest.approx(1 / 365)


def test_longest_gap_includes_leading_and_trailing_stretches():
    days = pd.DatetimeIndex(["2018-01-11", "2018-01-12", "2018-01-20"])
    assert coverage.longest_gap_days(days, "2018-01-01", "2018-01-31") == 11
    assert coverage.longest_interior_gap_days(days, "2018-01-01", "2018-01-31") == 7


def test_longest_gap_without_data_is_whole_window():
    empty = pd.DatetimeIndex([])
    assert coverage.longest_gap_days(empty, "2018-01-01", "2018-01-10") == 10
    assert coverage.longest_interior_gap_days(empty, "2018-01-01", "2018-01-10") == 0


def test_turbidity_statistics_use_daily_means_and_daily_maxima():
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "mean": [10.0, 20.0, 300.0, 400.0],
            "maximum": [15.0, 25.0, 500.0, np.nan],
        }
    )
    result = coverage.turbidity_statistics(daily, 200.0)
    assert result["turbidity_median"] == pytest.approx(160.0)
    assert result["turbidity_max"] == 500.0
    assert result["turbidity_share_above_200"] == 0.5


def test_turbidity_statistics_threshold_is_strict():
    daily = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=2), "mean": [200.0, 200.1], "maximum": [1, 1]}
    )
    assert coverage.turbidity_statistics(daily, 200.0)["turbidity_share_above_200"] == 0.5


def test_turbidity_statistics_empty_is_nan():
    result = coverage.turbidity_statistics(pd.DataFrame(columns=coverage.DAILY_COLUMNS), 200.0)
    assert all(np.isnan(value) for value in result.values())


def test_fallback_windows_and_sampled_coverage():
    windows = coverage.fallback_windows([2024, 2025], ["02-01", "08-01"], 7)
    assert len(windows) == 4
    assert windows[0] == ("2024-02-01", "2024-02-07")
    days = pd.date_range("2024-02-01", periods=7).append(pd.date_range("2025-08-01", periods=7))
    result = coverage.sampled_coverage(days, windows, [2024, 2025])
    assert result[2024] == 0.5
    assert result[2025] == 0.5


def test_daily_from_instantaneous_aggregates_per_day_and_drops_missing():
    index = pd.DatetimeIndex(["2024-05-01 00:00", "2024-05-01 00:15", "2024-05-02 00:00"], tz="UTC")
    frame = pd.DataFrame({"63680": [1.0, 3.0, np.nan], "63680_cd": ["A", "A", "A"]}, index=index)
    result = coverage.daily_from_instantaneous(frame, "63680")
    assert list(result["date"]) == [pd.Timestamp("2024-05-01")]
    assert list(result["mean"]) == [2.0]
    assert list(result["maximum"]) == [3.0]


def test_daily_from_instantaneous_empty_input():
    assert coverage.daily_from_instantaneous(None, "63680").empty
    assert coverage.daily_from_instantaneous(pd.DataFrame(), "63680").empty


TAIL_SETTINGS = {
    "start": "2024-01-01",
    "end": "2025-12-31",
    "tail_margin_days": 31,
    "fallback_window_days": 7,
    "fallback_window_starts": ["02-01", "08-01"],
}


def test_tail_windows_only_after_series_end_and_none_when_current():
    ended = pd.Timestamp("2024-06-30")
    windows = coverage.tail_windows(TAIL_SETTINGS, ended)
    assert [pair[0] for pair in windows] == ["2024-08-01", "2025-02-01", "2025-08-01"]
    assert coverage.tail_windows(TAIL_SETTINGS, pd.Timestamp("2025-12-15")) == []


def test_tail_adjusted_coverage_estimates_missing_years_from_samples():
    days = pd.date_range("2024-01-01", "2024-06-30")
    content = pd.DataFrame({"date": days, "mean": 1.0, "maximum": 1.0})
    tail_days = pd.date_range("2025-02-01", periods=7).append(
        pd.date_range("2025-08-01", periods=3)
    )
    tail = pd.DataFrame({"date": tail_days, "mean": 1.0, "maximum": 1.0})
    result = coverage.tail_adjusted_coverage(content, tail, TAIL_SETTINGS)
    assert result[2024] == pytest.approx(182 / 366)
    assert result[2025] == pytest.approx(10 / 14)


def test_estimated_years_for_sample_only_and_truncated_series():
    years = [2024, 2025]
    assert coverage.estimated_years("iv_sample", pd.DataFrame(), TAIL_SETTINGS) == years
    ended = pd.DataFrame({"date": pd.date_range("2024-01-01", "2024-06-30")})
    assert coverage.estimated_years("daily", ended, TAIL_SETTINGS) == years
    current = pd.DataFrame({"date": pd.date_range("2024-01-01", "2025-12-20")})
    assert coverage.estimated_years("daily", current, TAIL_SETTINGS) == []
    assert coverage.estimated_years("missing", pd.DataFrame(), TAIL_SETTINGS) == []
