from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rivwq.usgs import scout

FIXTURES = Path(__file__).parent / "fixtures" / "usgs_scout"
TEXT = {"site_no": str, "parm_cd": str, "huc_cd": str, "parameter_code": str}
REFERENCES = ["06888990", "06887500", "06892350", "06892518"]


@pytest.fixture(scope="module")
def config():
    return scout.load_config()


@pytest.fixture(scope="module")
def parameters():
    return pd.read_csv(FIXTURES / "parameter_codes.csv", dtype=TEXT)


@pytest.fixture(scope="module")
def table(config, parameters):
    catalog = pd.read_csv(FIXTURES / "series_catalog.csv", dtype=TEXT)
    details = pd.read_csv(FIXTURES / "site_details.csv", dtype=TEXT)
    return scout.build_tables(config, parameters, catalog, details)


def test_turbidity_codes_are_fnu_only(config, parameters):
    codes = scout.resolve_all_codes(parameters, config)["turbidity"]
    assert set(codes["parameter_code"]) == {"32265", "63680"}
    assert set(codes["unit_of_measure"]) == {"FNU"}


def test_other_variables_resolve_expected_codes(config, parameters):
    codes = scout.resolve_all_codes(parameters, config)
    assert list(codes["dissolved_oxygen"]["parameter_code"]) == ["00300"]
    assert list(codes["water_temperature"]["parameter_code"]) == ["00010"]
    assert "00060" in set(codes["discharge"]["parameter_code"])
    assert "00065" in set(codes["gage_height"]["parameter_code"])
    assert "32316" in set(codes["chlorophyll_fluorescence"]["parameter_code"])
    assert "32319" in set(codes["phycocyanin"]["parameter_code"])


def test_tidal_codes_found(config, parameters):
    assert "00067" in scout.tidal_codes(parameters, config)


def test_references_pass_and_report_fluorescence(table):
    for site in REFERENCES:
        assert table.at[site, "meets_criteria"]
        assert table.at[site, "fluorescence_present"]
        assert table.at[site, "is_reference"]


def test_topeka_series_summary(table):
    row = table.loc["06888990"]
    assert row["turbidity_code"] == "63680"
    assert row["turbidity_first"] == pd.Timestamp("2018-11-15")
    assert row["state"] == "KS"
    assert row["huc8"] == "10270102"
    assert row["drainage_area_sqmi"] == 56686.0
    assert 0.99 < row["turbidity_catalog_day_fraction"] <= 1.0


def test_overlap_is_clipped_to_sentinel_start(table):
    assert table.at["06887500", "overlap_start"] == pd.Timestamp("2018-01-01")
    assert table.at["06888990", "overlap_start"] == pd.Timestamp("2018-11-15")


def test_failure_reasons(table):
    assert table.at["01017060", "fail_reasons"] == "missing_required_series"
    assert table.at["01104430", "fail_reasons"] == "record_ended_early"
    assert table.at["01308000", "fail_reasons"] == "overlap_too_short"
    assert table.at["01467200", "fail_reasons"] == "tidal_or_canal"


def test_funnel_counts(table):
    counts = scout.funnel(table)
    assert counts["screened_sites"] == 9
    assert counts["meets_criteria"] == 5
    assert counts["tidal_or_canal"] == 1


def test_day_fraction_never_exceeds_one(table):
    fractions = table.filter(like="_day_fraction").astype(float).stack().dropna()
    assert (fractions <= 1.0).all()


def test_median_of_daily_medians():
    stats = pd.DataFrame({"p50_va": [1.0, 3.0, 100.0, np.nan]})
    assert scout.median_of_daily_medians(stats) == 3.0
    assert np.isnan(scout.median_of_daily_medians(stats.iloc[0:0]))


def test_seasonal_peak_month():
    stats = pd.DataFrame(
        {"month_nu": [4, 5, 5, 6, 6], "mean_va": [10, 50, 70, 20, 30]},
    )
    assert scout.seasonal_peak_month(stats) == 5.0


def test_regulated_peak_share():
    peaks = pd.DataFrame({"peak_cd": ["6", np.nan, "2,6", "7"]})
    assert scout.regulated_peak_share(peaks, ["5", "6"]) == 0.5
    assert np.isnan(scout.regulated_peak_share(peaks.iloc[0:0], ["5", "6"]))


def test_classifiers(config):
    strata = config["strata"]
    assert scout.classify_turbidity(150.0, strata) == "turbid"
    assert scout.classify_turbidity(3.0, strata) == "clear"
    assert scout.classify_turbidity(12.0, strata) == "intermediate"
    assert scout.classify_turbidity(np.nan, strata) == "unknown"
    assert scout.classify_regulation(0.9, strata) == "regulated"
    assert scout.classify_regulation(0.0, strata) == "free_flowing"
    assert scout.classify_regulation(0.3, strata) == "mixed"
    assert scout.classify_regime(6, 5000.0, 45.0, strata) == "snowmelt"
    assert scout.classify_regime(6, 800.0, 39.0, strata) == "rain_fed"
    assert scout.classify_regime(4, 800.0, 45.0, strata) == "snowmelt"
    assert scout.classify_cloudiness("KS", strata) == "sunnier"
    assert scout.classify_cloudiness("WA", strata) == "cloudier"
    assert scout.classify_cloudiness("ZZ", strata) == "unknown"


def test_chunked_covers_all_items_once():
    batches = list(scout.chunked(list(range(7)), 3))
    assert batches == [[0, 1, 2], [3, 4, 5], [6]]


def synthetic_candidates(count: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for index in range(count):
        rows.append(
            {
                "site_no": f"{index:08d}",
                "dec_lat_va": 30.0 + (index % 10) * 2.0,
                "dec_long_va": -120.0 + (index // 10) * 3.0,
                "turbidity_class": ["turbid", "clear"][index % 2],
                "regulation_class": ["regulated", "free_flowing"][(index // 2) % 2],
                "regime_class": ["rain_fed", "snowmelt"][(index // 4) % 2],
                "cloud_class": ["sunnier", "cloudier"][(index // 8) % 2],
                "overlap_years": 5.0 + rng.random() * 3,
                "min_required_catalog_day_fraction": 0.7 + rng.random() * 0.3,
                "fluorescence_present": bool(index % 3 == 0),
                "drainage_area_sqmi": 1000.0 + index,
                "sampled_day_fraction": 0.95,
                "meets_criteria": True,
                "is_reference": index == 0,
            }
        )
    return pd.DataFrame(rows).set_index("site_no", drop=False)


def test_shortlist_size_separation_and_coverage(config):
    candidates = synthetic_candidates(60)
    settings = config["shortlist"]
    shortlist = scout.select_shortlist(candidates, settings)
    assert len(shortlist) == settings["size"]
    assert "00000000" not in shortlist.index
    for axis in scout.STRATUM_AXES:
        assert shortlist[axis].nunique() == 2
    positions = shortlist[["dec_lat_va", "dec_long_va"]].to_numpy()
    for first in range(len(positions)):
        for second in range(first + 1, len(positions)):
            distance = scout.haversine_km(*positions[first], *positions[second])
            assert distance >= settings["min_separation_km"]
    assert shortlist["reason"].str.len().min() > 0


def test_shortlist_is_deterministic(config):
    candidates = synthetic_candidates(60)
    first = scout.select_shortlist(candidates, config["shortlist"])
    second = scout.select_shortlist(candidates, config["shortlist"])
    assert list(first.index) == list(second.index)


def test_shortlist_ignores_failing_sites(config):
    candidates = synthetic_candidates(60)
    candidates.loc[candidates.index[1:30], "meets_criteria"] = False
    shortlist = scout.select_shortlist(candidates, config["shortlist"])
    assert shortlist["meets_criteria"].all()


def test_haversine_known_distance():
    distance = scout.haversine_km(39.0, -95.7, 39.0, -94.7)
    assert distance == pytest.approx(86.5, abs=1.0)


def test_shortlist_requires_known_strata_and_minimum_area(config):
    candidates = synthetic_candidates(60)
    candidates.loc[candidates.index[1:20], "turbidity_class"] = "unknown"
    candidates.loc[candidates.index[20:40], "drainage_area_sqmi"] = 10.0
    shortlist = scout.select_shortlist(candidates, config["shortlist"])
    assert (shortlist["turbidity_class"] != "unknown").all()
    assert (shortlist["drainage_area_sqmi"] >= config["shortlist"]["min_drainage_area_sqmi"]).all()


def test_median_of_instantaneous_values_pools_windows():
    first = pd.DataFrame({"63680": [1.0, 2.0, 3.0]})
    second = pd.DataFrame({"63680": [10.0, np.nan], "63680_cd": ["A", "A"]})
    assert scout.median_of_instantaneous_values([first, second], "63680") == 2.5
    assert np.isnan(scout.median_of_instantaneous_values([], "63680"))
    assert np.isnan(scout.median_of_instantaneous_values([first], "00300"))


def test_day_fraction_counts_distinct_days_with_values():
    index = pd.to_datetime(
        ["2025-01-01 00:00", "2025-01-01 00:15", "2025-01-03 12:00", "2025-01-04 00:00"]
    )
    frame = pd.DataFrame({"63680": [1.0, 2.0, 3.0, np.nan]}, index=index)
    fraction = scout.day_fraction_from_instantaneous(frame, "63680", "2025-01-01", "2025-01-10")
    assert fraction == pytest.approx(0.2)
    assert (
        scout.day_fraction_from_instantaneous(frame.iloc[0:0], "63680", "2025-01-01", "2025-01-10")
        == 0.0
    )
    assert scout.day_fraction_from_instantaneous(None, "63680", "2025-01-01", "2025-01-10") == 0.0


def test_shortlist_excludes_gappy_series(config):
    candidates = synthetic_candidates(60)
    candidates.loc[candidates.index[1:30], "sampled_day_fraction"] = 0.3
    shortlist = scout.select_shortlist(candidates, config["shortlist"])
    assert (
        shortlist["sampled_day_fraction"] >= config["shortlist"]["min_sampled_day_fraction"]
    ).all()


def test_sensor_suffixed_columns_are_used():
    index = pd.to_datetime(["2025-01-01 00:00", "2025-01-02 00:00"])
    frame = pd.DataFrame(
        {"63680_ysi exo": [4.0, 6.0], "63680_ysi exo_cd": ["A", "A"], "site_no": ["1", "1"]},
        index=index,
    )
    assert scout.value_columns(frame, "63680") == ["63680_ysi exo"]
    assert scout.median_of_instantaneous_values([frame], "63680") == 5.0
    assert scout.day_fraction_from_instantaneous(
        frame, "63680", "2025-01-01", "2025-01-04"
    ) == pytest.approx(0.5)
