import numpy as np
import pandas as pd
import pytest

from rivwq.gee import screening, width_cloud

SETTINGS = {
    "circle_radius_m": 2000.0,
    "occurrence_thresholds": [25, 50, 75],
    "primary_occurrence_threshold": 50,
    "lake_like_min_water_share": 0.4,
    "pixel_size_m": 10.0,
    "min_pixels_across": 5.0,
}
CIRCLE_AREA = np.pi * 2000.0**2


def raw(width: float, near: float = 90.0) -> dict:
    return {
        "total_area_m2": CIRCLE_AREA,
        "water_area_m2_occ25": width * 4000.0 * 1.2,
        "water_area_m2_occ50": width * 4000.0,
        "water_area_m2_occ75": width * 4000.0 * 0.8,
        "max_occurrence_near_gauge": near,
    }


def test_width_is_area_over_diameter_with_sensitivity():
    result = screening.width_metrics(raw(100.0), SETTINGS)
    assert result["width_m"] == pytest.approx(100.0)
    assert result["width_m_occ25"] == pytest.approx(120.0)
    assert result["width_m_occ75"] == pytest.approx(80.0)
    assert result["pixels_across"] == pytest.approx(10.0)
    assert result["water_share"] == pytest.approx(100.0 * 4000.0 / CIRCLE_AREA)


def test_pixels_across_threshold_is_strict():
    assert screening.width_metrics(raw(49.0), SETTINGS)["under_min_pixels_across"]
    assert not screening.width_metrics(raw(50.0), SETTINGS)["under_min_pixels_across"]


def test_lake_like_rule_uses_water_share():
    assert not screening.is_lake_like(0.39, 0.4)
    assert screening.is_lake_like(0.4, 0.4)
    lake = {**raw(100.0), "water_area_m2_occ50": 0.6 * CIRCLE_AREA}
    assert screening.width_metrics(lake, SETTINGS)["lake_like"]


def test_gauge_near_water_requires_threshold_occurrence():
    assert screening.gauge_near_water(50.0, 50)
    assert not screening.gauge_near_water(49.0, 50)
    assert not screening.gauge_near_water(float("nan"), 50)
    assert not screening.width_metrics(raw(100.0, near=0.0), SETTINGS)["gauge_near_water"]


def per_date_frame() -> pd.DataFrame:
    rows = [
        ("s1", "2018-03-01", 0.9, 0.5, 1.0),
        ("s1", "2018-03-06", 0.8, 0.9, 1.0),
        ("s1", "2018-03-11", 0.79, 0.1, 1.0),
        ("s1", "2019-03-01", 1.0, 1.0, 0.0),
    ]
    return pd.DataFrame(rows, columns=["site_no", "date", "valid_cs", "valid_scl", "covered"])


def test_cloud_summary_counts_dates_and_reports_masks_separately():
    summary = screening.cloud_summary(per_date_frame(), ["s1"], [2018, 2019, 2020], 0.8)
    row = summary.loc["s1"]
    assert row["dates_2018"] == 3
    assert row["dates_2019"] == 1
    assert row["dates_2020"] == 0
    assert row["dates_total"] == 4
    assert row["dates_per_year_mean"] == pytest.approx(4 / 3)
    assert row["cloud_score_share_2018"] == pytest.approx(2 / 3)
    assert row["scl_share_2018"] == pytest.approx(1 / 3)
    assert row["cloud_score_share"] == pytest.approx(3 / 4)
    assert row["scl_share"] == pytest.approx(2 / 4)
    assert np.isnan(row["cloud_score_share_2020"])
    assert row["dates_with_roi_data_share"] == pytest.approx(3 / 4)


def test_valid_fraction_exactly_at_minimum_counts_as_valid():
    frame = pd.DataFrame({"valid_cs": [0.8]})
    assert screening.valid_date_share(frame, "valid_cs", 0.8) == 1.0


def candidates_frame() -> pd.DataFrame:
    columns = {name: [None] * 4 for name in screening.BASE_COLUMNS}
    return pd.DataFrame(columns, index=pd.Index(["a", "b", "c", "d"], name="site_no"))


def test_select_sites_assigns_roles_and_rejects_unknown_sites():
    candidates = candidates_frame()
    shortlist = candidates.loc[["a"]]
    sites = screening.select_sites(shortlist, candidates, ["d"], ["b"])
    assert dict(sites["role"]) == {"a": "shortlist", "b": "extra", "d": "reference"}
    with pytest.raises(KeyError):
        screening.select_sites(shortlist, candidates, ["zzz"], [])


def test_assemble_sorts_by_drainage_area():
    sites = pd.DataFrame({"drainage_area_sqmi": [30.0, 10.0, 20.0]}, index=["x", "y", "z"])
    table = screening.assemble(
        sites, pd.DataFrame(index=sites.index), pd.DataFrame(index=sites.index)
    )
    assert list(table.index) == ["y", "z", "x"]


def test_pending_jobs_skips_cached_site_years(tmp_path):
    sites = pd.DataFrame(index=["a", "b"])
    (tmp_path / "cloud").mkdir()
    (tmp_path / "cloud" / "a_2018.csv").write_text("x")
    jobs = width_cloud.pending_jobs(sites, [2018, 2019], tmp_path)
    assert jobs == [("a", 2019), ("b", 2018), ("b", 2019)]


def test_config_thresholds_are_present_and_consistent():
    config = width_cloud.load_config()
    assert (
        config["width"]["primary_occurrence_threshold"] in config["width"]["occurrence_thresholds"]
    )
    assert 0.0 < config["cloud"]["min_valid_roi_fraction"] <= 1.0
    assert set(config["datasets"]) == {"surface_reflectance", "cloud_score", "surface_water"}
