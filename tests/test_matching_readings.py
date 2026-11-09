import pandas as pd
import pytest

from rivwq.matching import readings
from rivwq.usgs import qc_rules
from tests.support_matching import regular, tidy_series

CONFIG = readings.load_config()
QC = qc_rules.load_config()


def test_native_interval_is_median_step_per_year():
    early = regular("2024-06-01", 200, "15min")
    late = regular("2025-06-01", 100, "60min")
    table = readings.native_interval_table(pd.concat([early, late], ignore_index=True))
    assert table.set_index("year")["native_minutes"].to_dict() == {2024: 15, 2025: 60}


def test_native_interval_handles_30_minute_series():
    table = readings.native_interval_table(regular("2025-01-01", 100, "30min"))
    assert table["native_minutes"].tolist() == [30]


def test_native_interval_ignores_missing_values():
    series = regular("2025-01-01", 20, "15min")
    series.loc[5:9, "value"] = float("nan")
    table = readings.native_interval_table(series)
    assert table["native_minutes"].tolist() == [15]


def test_naive_timestamps_are_rejected():
    series = regular("2025-01-01", 5, "15min")
    series["datetime_utc"] = series["datetime_utc"].dt.tz_localize(None)
    with pytest.raises(ValueError):
        readings.native_interval_table(series)


def test_non_utc_timestamps_are_rejected():
    series = regular("2025-01-01", 5, "15min")
    series["datetime_utc"] = series["datetime_utc"].dt.tz_convert("America/Chicago")
    with pytest.raises(ValueError):
        readings.to_ns(series["datetime_utc"])


def test_to_ns_is_unit_independent():
    microseconds = pd.Series(pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True)).astype(
        "datetime64[us, UTC]"
    )
    nanoseconds = microseconds.astype("datetime64[ns, UTC]")
    assert readings.to_ns(microseconds).tolist() == readings.to_ns(nanoseconds).tolist()


def test_qc_excluded_rows_are_removed_and_counted_in_nothing_else():
    series = regular("2025-01-01", 10, "15min")
    series.loc[3, "qualifiers"] = "EQUIP"
    series.loc[4, "value"] = float("nan")
    built = readings.build_series(series, CONFIG, QC, "standard", "63680")
    assert len(built) == 1
    assert len(built[0].values) == 8


def test_estimated_and_provisional_flags_are_carried():
    series = regular("2025-01-01", 4, "15min", code="00060")
    series.loc[1, "qualifiers"] = "ESTIMATED"
    series.loc[2, "approval_status"] = "Provisional"
    built = readings.build_series(series, CONFIG, QC, "standard", "00060")[0]
    assert built.estimated.tolist() == [False, True, False, False]
    assert built.provisional.tolist() == [False, False, True, False]


def test_estimated_is_not_a_qc_exclusion_at_any_level():
    series = regular("2025-01-01", 4, "15min", code="00060", qualifiers="ESTIMATED")
    for level in ("lenient", "standard", "strict"):
        built = readings.build_series(series, CONFIG, QC, level, "00060")[0]
        assert len(built.values) == 4


def test_levels_differ_for_flatline_only_at_strict():
    series = regular("2025-01-01", 4 * 24 * 3, "15min", code="00300", values=[7.0] * (4 * 24 * 3))
    standard = readings.build_series(series, CONFIG, QC, "standard", "00300")
    strict = readings.build_series(series, CONFIG, QC, "strict", "00300")
    assert len(standard[0].values) == 4 * 24 * 3
    assert strict == []


def test_exact_duplicates_are_dropped_conflicting_kept():
    series = regular("2025-01-01", 3, "15min", values=[1.0, 2.0, 3.0])
    duplicate = series.iloc[[1]].copy()
    conflicting = series.iloc[[2]].copy()
    conflicting["value"] = 9.0
    combined = pd.concat([series, duplicate, conflicting], ignore_index=True)
    built = readings.build_series(combined, CONFIG, QC, "standard", "63680")[0]
    assert sorted(built.values.tolist()) == [1.0, 2.0, 3.0, 9.0]


def test_two_sensors_become_two_series_with_own_native_interval():
    first = regular("2025-01-01", 100, "15min", sensor="a")
    second = regular("2025-03-01", 100, "60min", sensor="b")
    built = readings.build_series(
        pd.concat([first, second], ignore_index=True), CONFIG, QC, "standard", "63680"
    )
    assert {item.sensor_id: item.native_overall for item in built} == {"a": 15.0, "b": 60.0}


def test_missing_parameter_gives_no_series():
    assert (
        readings.build_series(regular("2025-01-01", 5, "15min"), CONFIG, QC, "standard", "00300")
        == []
    )


def test_native_for_unseen_year_falls_back_to_overall():
    built = readings.build_series(
        regular("2025-01-01", 50, "30min"), CONFIG, QC, "standard", "63680"
    )[0]
    assert built.native_minutes(2025) == 30.0
    assert built.native_minutes(2030) == 30.0


def test_tidy_series_helper_accepts_naive_stamps():
    assert tidy_series(["2025-01-01 00:00"]).iloc[0]["datetime_utc"].tz is not None


def test_single_reading_year_has_no_native_interval_and_falls_back():
    long = regular("2024-06-01", 100, "30min")
    single = regular("2025-06-01", 1, "30min")
    built = readings.build_series(pd.concat([long, single]), CONFIG, QC, "standard", "63680")[0]
    assert built.native_minutes(2025) == 30.0
