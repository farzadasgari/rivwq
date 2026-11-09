import numpy as np
import pandas as pd
import pytest

from rivwq.matching import readings, windows
from rivwq.usgs import qc_rules
from tests.support_matching import regular, scene_times, tidy_series

CONFIG = readings.load_config()
QC = qc_rules.load_config()


def series_of(frame, code="63680"):
    return readings.build_series(frame, CONFIG, QC, "standard", code)


def match(frame, *stamps, code="63680"):
    result = windows.match_parameter(series_of(frame, code), scene_times(*stamps), CONFIG)
    return result.set_index(["scene_pos", "variant"])


def test_variants_are_the_five_registered_ones():
    assert windows.window_variants(CONFIG) == [
        "nearest",
        "median_15",
        "median_30",
        "median_60",
        "daily_mean",
    ]


def test_fifteen_minute_series_nearest_within_half_interval():
    frame = regular("2025-03-01 10:00", 8, "15min", values=range(8))
    result = match(frame, "2025-03-01 10:23")
    row = result.loc[(0, "nearest")]
    assert row["value"] == 2.0
    assert abs(row["dt_min"]) <= 7.5


def test_fifteen_minute_nearest_rejects_beyond_half_interval():
    early = regular("2025-03-01 10:00", 4, "15min")
    late = regular("2025-03-01 12:00", 4, "15min")
    frame = pd.concat([early, late], ignore_index=True)
    result = match(frame, "2025-03-01 11:00")
    assert result.loc[(0, "nearest"), "n"] == 0
    assert np.isnan(result.loc[(0, "nearest"), "value"])


def test_nearest_tie_takes_the_earlier_reading():
    frame = regular("2025-03-01 10:00", 3, "15min", values=[10.0, 20.0, 30.0])
    result = match(frame, "2025-03-01 10:07:30")
    row = result.loc[(0, "nearest")]
    assert row["value"] == 10.0
    assert row["dt_min"] == -7.5


def test_thirty_minute_series_nearest_uses_fifteen_minute_tolerance():
    frame = regular("2025-03-01 10:00", 6, "30min", values=range(6))
    inside = match(frame, "2025-03-01 10:44")
    outside = match(frame, "2025-03-01 10:50")
    assert inside.loc[(0, "nearest"), "value"] == 1.0
    assert outside.loc[(0, "nearest"), "value"] == 2.0
    sparse = frame.drop(index=[1, 2]).reset_index(drop=True)
    gap = match(sparse, "2025-03-01 10:45")
    assert gap.loc[(0, "nearest"), "n"] == 0


def test_sixty_minute_series_windows_as_written():
    frame = regular("2025-03-01 09:00", 5, "60min", values=[0.0, 10.0, 20.0, 30.0, 40.0])
    result = match(frame, "2025-03-01 10:40")
    assert result.loc[(0, "nearest"), "value"] == 20.0
    assert result.loc[(0, "median_15"), "n"] == 0
    assert result.loc[(0, "median_30"), "n"] == 1
    assert result.loc[(0, "median_30"), "value"] == 20.0
    assert result.loc[(0, "median_60"), "n"] == 2
    assert result.loc[(0, "median_60"), "value"] == 15.0


def test_window_ends_are_inclusive():
    frame = regular("2025-03-01 10:00", 5, "15min", values=[0.0, 1.0, 2.0, 3.0, 4.0])
    result = match(frame, "2025-03-01 10:30")
    assert result.loc[(0, "median_15"), "n"] == 3
    assert result.loc[(0, "median_30"), "n"] == 5


def test_median_is_used_in_windows_not_mean():
    times = pd.to_datetime(["2025-03-01 10:00", "2025-03-01 10:10", "2025-03-01 10:15"], utc=True)
    frame = tidy_series(times, values=[1.0, 2.0, 100.0])
    result = match(frame, "2025-03-01 10:10")
    assert result.loc[(0, "median_15"), "value"] == 2.0


def test_daily_mean_uses_the_utc_day_of_the_scene():
    times = pd.date_range("2025-03-01 00:00", "2025-03-02 23:45", freq="15min", tz="UTC")
    values = np.where(times.day == 1, 10.0, 50.0)
    frame = tidy_series(times, values=values)
    result = match(frame, "2025-03-01 23:50", "2025-03-02 00:05")
    assert result.loc[(0, "daily_mean"), "value"] == 10.0
    assert result.loc[(0, "daily_mean"), "n"] == 96
    assert result.loc[(1, "daily_mean"), "value"] == 50.0


def test_midnight_edge_window_reaches_into_next_day_but_daily_does_not():
    times = pd.to_datetime(["2025-03-01 23:45", "2025-03-02 00:00", "2025-03-02 00:15"], utc=True)
    frame = tidy_series(times, values=[1.0, 2.0, 3.0])
    result = match(frame, "2025-03-01 23:55")
    assert result.loc[(0, "median_15"), "n"] == 2
    assert result.loc[(0, "daily_mean"), "n"] == 1
    assert result.loc[(0, "daily_mean"), "value"] == 1.0


def test_non_utc_scene_times_are_rejected():
    frame = regular("2025-03-01 10:00", 4, "15min")
    local = scene_times("2025-03-01 10:00").dt.tz_convert("America/Chicago")
    with pytest.raises(ValueError):
        windows.match_parameter(series_of(frame), local, CONFIG)


def test_empty_series_gives_rows_with_zero_readings_for_every_variant():
    result = windows.match_parameter([], scene_times("2025-03-01 10:00"), CONFIG)
    assert len(result) == 5
    assert (result["n"] == 0).all()
    assert result["value"].isna().all()


def test_no_value_is_interpolated_between_readings():
    frame = regular("2025-03-01 10:00", 2, "60min", values=[0.0, 100.0])
    result = match(frame, "2025-03-01 10:30")
    assert result.loc[(0, "median_15"), "n"] == 0
    assert result.loc[(0, "nearest"), "value"] in (0.0, 100.0)


def test_two_sensors_prefer_more_readings_and_flag_the_conflict():
    first = regular("2025-03-01 10:00", 8, "15min", sensor="a", values=[1.0] * 8)
    second = regular("2025-03-01 10:45", 2, "15min", sensor="b", values=[9.0] * 2)
    frame = pd.concat([first, second], ignore_index=True)
    result = match(frame, "2025-03-01 11:00")
    row = result.loc[(0, "median_30")]
    assert row["sensor_id"] == "a"
    assert row["value"] == 1.0
    assert bool(row["other_sensor_in_window"]) is True


def test_two_sensors_never_overlapping_report_no_conflict():
    first = regular("2025-03-01 00:00", 8, "15min", sensor="a", values=[1.0] * 8)
    second = regular("2025-06-01 00:00", 8, "15min", sensor="b", values=[9.0] * 8)
    frame = pd.concat([first, second], ignore_index=True)
    result = match(frame, "2025-03-01 01:00", "2025-06-01 01:00")
    assert result.loc[(0, "nearest"), "sensor_id"] == "a"
    assert result.loc[(1, "nearest"), "sensor_id"] == "b"
    assert not result["other_sensor_in_window"].any()


def test_empty_window_has_no_sensor_id():
    first = regular("2025-03-01 00:00", 4, "15min", sensor="a")
    second = regular("2025-06-01 00:00", 4, "15min", sensor="b")
    frame = pd.concat([first, second], ignore_index=True)
    result = match(frame, "2025-04-15 12:00")
    assert result.loc[(0, "median_60")].isna()["sensor_id"]


def test_two_sensors_tie_on_count_uses_nearer_reading():
    first = regular("2025-03-01 10:00", 2, "15min", sensor="a", values=[1.0, 1.0])
    second = regular("2025-03-01 10:10", 2, "15min", sensor="b", values=[2.0, 2.0])
    frame = pd.concat([first, second], ignore_index=True)
    result = match(frame, "2025-03-01 10:12")
    assert result.loc[(0, "median_15"), "n"] == 2
    assert result.loc[(0, "median_15"), "sensor_id"] == "b"


def test_provisional_and_estimated_shares_in_window():
    frame = regular("2025-03-01 10:00", 4, "15min", code="00060", values=[1, 2, 3, 4])
    frame.loc[0, "approval_status"] = "Provisional"
    frame.loc[[1, 2], "qualifiers"] = "ESTIMATED"
    result = match(frame, "2025-03-01 10:22", code="00060")
    row = result.loc[(0, "median_30")]
    assert row["provisional_share"] == pytest.approx(0.25)
    assert row["estimated_share"] == pytest.approx(0.5)


def test_native_interval_follows_the_scene_year():
    early = regular("2024-06-01 00:00", 200, "15min")
    late = regular("2025-06-01 00:00", 100, "60min")
    frame = pd.concat([early, late], ignore_index=True)
    result = match(frame, "2024-06-01 01:10", "2025-06-01 01:20")
    assert result.loc[(0, "nearest"), "native_min"] == 15
    assert result.loc[(1, "nearest"), "native_min"] == 60
    assert result.loc[(1, "nearest"), "n"] == 1
