import numpy as np
import pandas as pd
import pytest

from rivwq.usgs import tidy


def test_offset_strings_are_converted_across_spring_dst_change():
    local = pd.Series(
        ["2025-03-09T01:45:00-06:00", "2025-03-09T03:00:00-05:00", "2025-03-09T03:15:00-05:00"]
    )
    result = tidy.to_utc(local)
    assert list(result.dt.strftime("%Y-%m-%dT%H:%M")) == [
        "2025-03-09T07:45",
        "2025-03-09T08:00",
        "2025-03-09T08:15",
    ]
    assert result.diff().dropna().tolist() == [pd.Timedelta(minutes=15)] * 2


def test_offset_strings_keep_both_hours_of_autumn_dst_change():
    local = pd.Series(["2025-11-02T01:30:00-05:00", "2025-11-02T01:30:00-06:00"])
    result = tidy.to_utc(local)
    assert result.is_unique
    assert result.iloc[1] - result.iloc[0] == pd.Timedelta(hours=1)


def test_zone_aware_series_is_converted_to_utc():
    local = pd.Series(pd.to_datetime(["2025-07-01 12:00"])).dt.tz_localize("America/Chicago")
    assert tidy.to_utc(local).iloc[0] == pd.Timestamp("2025-07-01 17:00", tz="UTC")


def test_zone_aware_winter_and_summer_use_different_offsets():
    local = pd.Series(pd.to_datetime(["2025-01-15 12:00", "2025-07-15 12:00"]))
    local = local.dt.tz_localize("America/Chicago")
    assert list(tidy.to_utc(local).dt.hour) == [18, 17]


def test_naive_timestamps_are_rejected():
    with pytest.raises(ValueError):
        tidy.to_utc(pd.Series(pd.to_datetime(["2025-01-01 00:00"])))
    with pytest.raises(ValueError):
        tidy.to_utc(pd.Series(["2025-01-01T00:00:00"]))


def test_missing_timestamps_are_rejected():
    with pytest.raises(ValueError):
        tidy.to_utc(pd.Series(pd.to_datetime(["2025-01-01 00:00", None], utc=True)))


def test_mixed_fractional_second_strings_are_parsed():
    result = tidy.to_utc(
        pd.Series(["2020-03-01T00:09:59.067000+00:00", "2020-03-01T00:15:00+00:00"])
    )
    assert result.notna().all()
    assert result.iloc[0] == pd.Timestamp("2020-03-01 00:09:59.067", tz="UTC")


def test_utc_suffix_is_accepted():
    result = tidy.to_utc(pd.Series(["2025-01-01T00:15:00Z"]))
    assert result.iloc[0] == pd.Timestamp("2025-01-01 00:15", tz="UTC")


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, ""),
        (np.nan, ""),
        ([], ""),
        (["ESTIMATED"], "ESTIMATED"),
        (["e", "ICE"], "e|ICE"),
        (np.array(["A", "B"]), "A|B"),
        ("{e,ICE}", "e|ICE"),
        ("[ 'e', 'ICE' ]", "e|ICE"),
        ("", ""),
    ],
)
def test_qualifier_parsing(value, expected):
    assert tidy.parse_qualifiers(value) == expected


def response(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "monitoring_location_id",
            "time",
            "parameter_code",
            "time_series_id",
            "value",
            "qualifier",
            "approval_status",
            "unit_of_measure",
        ],
    )


def test_tidy_response_reshapes_and_sorts_without_dropping_rows():
    t0 = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    t1 = pd.Timestamp("2025-01-01 00:15", tz="UTC")
    frame = response(
        [
            ("USGS-0001", t1, "63680", "s1", 5.0, None, "Provisional", "_FNU"),
            ("USGS-0001", t0, "63680", "s1", np.nan, ["ICE"], "Approved", "_FNU"),
            ("USGS-0001", t0, "00300", "s2", 9.0, None, "Approved", "mg/l"),
            ("USGS-0001", t0, "00300", "s2", 9.0, None, "Approved", "mg/l"),
        ]
    )
    result = tidy.tidy_response(frame, "src")
    assert list(result.columns) == tidy.TIDY_COLUMNS
    assert len(result) == 4
    assert result["site_no"].unique().tolist() == ["0001"]
    assert result["parameter_code"].tolist() == ["00300", "00300", "63680", "63680"]
    turbidity = result[result["parameter_code"] == "63680"]
    assert turbidity["datetime_utc"].tolist() == [t0, t1]
    assert np.isnan(turbidity["value"].iloc[0])
    assert turbidity["qualifiers"].tolist() == ["ICE", ""]
    assert turbidity["approval_status"].tolist() == ["Approved", "Provisional"]
    assert set(result["source_column"]) == {"src"}


def test_two_sensors_of_one_parameter_stay_separate():
    t0 = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    frame = response(
        [
            ("USGS-0001", t0, "32316", "a", 1.0, None, "Approved", "ug/l"),
            ("USGS-0001", t0, "32316", "b", 2.0, None, "Approved", "ug/l"),
        ]
    )
    result = tidy.tidy_response(frame, "src")
    assert result["sensor_id"].tolist() == ["a", "b"]
    assert result["value"].tolist() == [1.0, 2.0]
