import numpy as np
import pandas as pd
import pytest

from rivwq.usgs import qc

SETTINGS = {
    "gap_days": 7.0,
    "flatline_hours": 24.0,
    "flatline_break_intervals": 4.0,
    "ceiling_near_fraction": 0.95,
    "sensor_ceilings": {"63680": 3900.0},
}
RUN_END = pd.Timestamp("2026-03-01", tz="UTC")


def build(times, values, code="63680", status="Approved", qualifiers="", unit="_FNU"):
    return pd.DataFrame(
        {
            "site_no": "0001",
            "datetime_utc": pd.DatetimeIndex(times),
            "parameter_code": code,
            "sensor_id": "s1",
            "value": values,
            "qualifiers": qualifiers,
            "approval_status": status,
            "unit_of_measure": unit,
            "source_column": "src",
        }
    )


def quarter_hours(start: str, count: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=count, freq="15min", tz="UTC")


def test_median_interval_ignores_duplicates():
    times = pd.Series(list(quarter_hours("2025-01-01", 5)) + [pd.Timestamp("2025-01-01", tz="UTC")])
    assert qc.median_interval_seconds(times) == 900.0


def test_duplicates_are_counted_and_conflicts_separated():
    times = quarter_hours("2025-01-01", 3).append(quarter_hours("2025-01-01", 2))
    values = [1.0, 2.0, 3.0, 1.0, 9.0]
    extra, conflicting = qc.duplicate_counts(build(times, values))
    assert extra == 2
    assert conflicting == 1


def test_gap_longer_than_limit_is_found_and_shorter_is_not():
    first = quarter_hours("2025-01-01", 4)
    after_five_days = quarter_hours("2025-01-06", 2)
    after_eight_days = quarter_hours("2025-01-14 12:00", 2)
    times = pd.Series(first.append(after_five_days).append(after_eight_days))
    gaps = qc.long_gaps(times, 7.0)
    assert len(gaps) == 1
    assert gaps["gap_days"].iloc[0] == pytest.approx(8.5, abs=0.1)


def test_flatline_over_a_day_is_flagged_and_shorter_run_is_not():
    times = quarter_hours("2025-01-01", 200)
    values = np.r_[np.full(120, 5.0), np.arange(60, dtype=float) + 10, np.full(20, 3.0)]
    runs = qc.flatline_runs(build(times, values), 900.0, 24.0, 4.0)
    assert len(runs) == 1
    assert runs["value"].iloc[0] == 5.0
    assert runs["run_hours"].iloc[0] == pytest.approx(29.75)


def test_flatline_is_broken_by_a_data_gap():
    first = quarter_hours("2025-01-01", 60)
    second = quarter_hours("2025-01-05", 60)
    times = first.append(second)
    runs = qc.flatline_runs(build(times, np.full(120, 7.0)), 900.0, 14.0, 4.0)
    assert len(runs) == 2


def test_nan_values_do_not_form_flatlines():
    times = quarter_hours("2025-01-01", 200)
    runs = qc.flatline_runs(build(times, np.full(200, np.nan)), 900.0, 24.0, 4.0)
    assert runs.empty


def test_ceiling_counts_at_and_near():
    values = pd.Series([100.0, 3700.0, 3830.0, 3900.0, 3990.0])
    assert qc.ceiling_counts(values, 3900.0, 0.95) == (2, 3)
    assert qc.ceiling_counts(values, None, 0.95) == (None, None)


def test_year_table_coverage_and_status_shares():
    times = quarter_hours("2025-12-01", 96 * 10)
    status = ["Provisional"] * 480 + ["Approved"] * 480
    frame = build(times, np.arange(960, dtype=float))
    frame["approval_status"] = status
    table = qc.year_table(frame, 900.0, RUN_END)
    row = table.iloc[0]
    assert row["year"] == 2025
    assert row["coverage_in_span"] == pytest.approx(1.0)
    assert row["coverage_calendar"] == pytest.approx(960 / (365 * 96))
    assert row["share_provisional"] == pytest.approx(0.5)
    assert row["share_approved"] == pytest.approx(0.5)
    assert row["share_other_status"] == 0.0


def test_series_summary_flags_negative_ceiling_units_and_qualifiers():
    times = quarter_hours("2025-01-01", 6)
    frame = build(times, [-0.5, 0.0, 3900.0, 4000.0, np.nan, 12.0])
    frame.loc[0, "qualifiers"] = "ICE"
    summary = qc.series_summary(frame, SETTINGS)
    assert summary["negative_values"] == 1
    assert summary["zero_values"] == 1
    assert summary["values_at_ceiling"] == 2
    assert summary["null_values"] == 1
    assert summary["qualified_values"] == 1
    assert summary["units_seen"] == "_FNU"
    assert summary["median_interval_minutes"] == 15.0
    assert summary["first_utc"] == times[0]


def test_qc_site_returns_one_row_per_series_and_never_changes_input():
    times = quarter_hours("2025-01-01", 10)
    frame = pd.concat(
        [build(times, np.arange(10.0)), build(times, np.arange(10.0), code="00300", unit="mg/l")]
    )
    before = frame.copy()
    result = qc.qc_site(frame, SETTINGS, RUN_END)
    assert sorted(result["summary"]["parameter_code"]) == ["00300", "63680"]
    assert result["years"]["parameter_code"].nunique() == 2
    pd.testing.assert_frame_equal(frame, before)
