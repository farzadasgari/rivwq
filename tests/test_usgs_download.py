import pandas as pd
import pytest

from rivwq.usgs import download

CONFIG = {
    "start_date": "2018-01-01",
    "catalog_computation": "Instantaneous",
    "fixed_parameters": {"63680": "turbidity", "00300": "dissolved_oxygen"},
    "fluorescence_name_pattern": "(?i)fchl|fpc|chlorophyll",
    "nominal_interval_seconds": 900,
}
RUN_END = pd.Timestamp("2026-03-01 12:00", tz="UTC")


def catalog(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "monitoring_location_id",
            "time_series_id",
            "parameter_code",
            "parameter_name",
            "computation_identifier",
            "begin",
            "end",
        ],
    )


def stamp(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


ROWS = [
    (
        "USGS-1",
        "a",
        "63680",
        "Turbidity, FNU",
        "Instantaneous",
        stamp("2015-01-01"),
        stamp("2026-03-01"),
    ),
    ("USGS-1", "b", "63680", "Turbidity, FNU", "Mean", stamp("2015-01-01"), stamp("2026-03-01")),
    (
        "USGS-1",
        "c",
        "32316",
        "fChl, water, in situ",
        "Instantaneous",
        stamp("2020-01-01"),
        stamp("2026-03-01"),
    ),
    (
        "USGS-1",
        "d",
        "32316",
        "fChl, water, in situ",
        "Instantaneous",
        stamp("2021-01-01"),
        stamp("2026-03-01"),
    ),
    ("USGS-1", "e", "99999", "Nitrate", "Instantaneous", stamp("2020-01-01"), stamp("2026-03-01")),
    (
        "USGS-1",
        "f",
        "00300",
        "Dissolved oxygen",
        "Instantaneous",
        stamp("2010-01-01"),
        stamp("2014-01-01"),
    ),
    (
        "USGS-2",
        "g",
        "63680",
        "Turbidity, FNU",
        "Instantaneous",
        stamp("2019-06-01"),
        stamp("2024-02-01"),
    ),
]


def test_select_series_keeps_instantaneous_fixed_and_fluorescence_only():
    selected = download.select_series(catalog(ROWS), CONFIG)
    assert selected["time_series_id"].tolist() == ["c", "d", "a", "g"]
    assert selected["site_no"].tolist() == ["1", "1", "1", "2"]


def test_missing_fixed_series_lists_absent_codes():
    selected = download.select_series(catalog(ROWS), CONFIG)
    missing = download.missing_fixed_series(selected, ["1", "2"], CONFIG)
    assert set(zip(missing["site_no"], missing["parameter_code"], strict=True)) == {
        ("1", "00300"),
        ("2", "00300"),
    }


def test_plan_covers_series_years_and_stops_at_run_end():
    selected = download.select_series(catalog(ROWS), CONFIG)
    plan = download.plan_site_years(selected, CONFIG, RUN_END)
    years = {site: [year for s, year, _ in plan if s == site] for site in ("1", "2")}
    assert years["1"] == list(range(2018, 2027))
    assert years["2"] == list(range(2019, 2025))
    codes = {site: codes for site, _, codes in plan}
    assert codes["1"] == ("32316", "63680")


def test_year_window_is_closed_before_next_year_and_capped_at_run_end():
    assert download.year_window(2024, RUN_END) == "2024-01-01T00:00:00Z/2024-12-31T23:59:59Z"
    assert download.year_window(2026, RUN_END) == "2026-01-01T00:00:00Z/2026-03-01T12:00:00Z"


def test_estimate_rows_clips_to_start_and_run_end():
    selected = download.select_series(catalog(ROWS[:1]), CONFIG)
    days = (RUN_END.normalize() - stamp("2018-01-01")).days
    expected = days * 96 + (12 * 4)
    assert download.estimate_rows(selected, CONFIG, RUN_END) == pytest.approx(expected, abs=96)


def test_with_backoff_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda seconds: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("busy")
        return "ok"

    assert download.with_backoff(flaky, attempts=5, base_seconds=1.0) == "ok"
    assert len(calls) == 3


def test_with_backoff_raises_after_last_attempt(monkeypatch):
    sleeps = []
    monkeypatch.setattr(download.time, "sleep", sleeps.append)

    def broken():
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        download.with_backoff(broken, attempts=3, base_seconds=2.0)
    assert sleeps == [2.0, 4.0]


def test_with_backoff_waits_out_quota_errors_without_using_normal_attempts(monkeypatch):
    sleeps = []
    monkeypatch.setattr(download.time, "sleep", sleeps.append)

    class QuotaExhausted(Exception):
        pass

    calls = []

    def limited():
        calls.append(1)
        if len(calls) < 4:
            raise QuotaExhausted("429")
        return "ok"

    result = download.with_backoff(
        limited, attempts=2, base_seconds=1.0, quota_wait_seconds=600.0, quota_attempts=5
    )
    assert result == "ok"
    assert sleeps == [600.0, 600.0, 600.0]
