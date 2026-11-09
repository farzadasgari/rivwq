import copy

import numpy as np
import pandas as pd
import yaml

from rivwq.matching import readings, table, windows
from rivwq.usgs import qc_rules
from tests.support_matching import regular

CONFIG = readings.load_config()
QC = qc_rules.load_config()
SITE = {
    "site_no": "0001",
    "layer": "core",
    "cv_group": "alpha",
    "targets": {"turbidity": True, "dissolved_oxygen": True, "fchl": False},
    "fchl_code": None,
}
ROIS = [("near_gauge", 0), ("reach", 0), ("reach", 1), ("extended", 0)]
CLOUDS = ["cs50", "scl"]


def gee_rows(site, stamps, spacecraft="S2A"):
    rows = []
    for number, stamp in enumerate(stamps):
        for roi, erosion in ROIS:
            for cloud in CLOUDS:
                rows.append(
                    {
                        "site_no": site,
                        "system_index": f"scene{number}",
                        "time_utc": pd.Timestamp(stamp, tz="UTC"),
                        "spacecraft": spacecraft,
                        "mgrs_tile": "15STD",
                        "roi": roi,
                        "erosion_px": erosion,
                        "cloud_variant": cloud,
                        "total_count": 100,
                        "valid_count": 0 if cloud == "scl" else 80,
                        "valid_fraction": 0.0 if cloud == "scl" else 0.8,
                    }
                )
    return pd.DataFrame(rows)


def usgs_frame():
    turbidity = regular("2025-03-01 00:00", 96 * 3, "15min", code="63680", values=range(288))
    oxygen = regular("2025-03-01 00:00", 96 * 3, "15min", code="00300", values=[8.0] * 288)
    discharge = regular("2025-03-01 00:00", 96 * 3, "15min", code="00060", values=[100.0] * 288)
    stage = regular("2025-03-01 00:00", 96 * 3, "15min", code="00065", values=[3.0] * 288)
    return pd.concat([turbidity, oxygen, discharge, stage], ignore_index=True)


def build(site=None, tidy=None, gee=None, invalid=None, level="standard"):
    return table.build_site(
        usgs_frame() if tidy is None else tidy,
        gee_rows("0001", ["2025-03-01 10:07", "2025-03-02 10:07"]) if gee is None else gee,
        SITE if site is None else site,
        {} if invalid is None else invalid,
        CONFIG,
        QC,
        level,
    )


def test_one_row_per_scene_roi_erosion_cloud_and_variant():
    result = build()["table"]
    assert len(result) == 2 * len(ROIS) * len(CLOUDS) * 5
    key = ["system_index", "roi", "erosion_px", "cloud_variant", "variant"]
    assert not result.duplicated(key).any()


def test_scene_in_usgs_gap_is_kept_with_empty_target():
    gee = gee_rows("0001", ["2025-03-01 10:07", "2025-03-20 10:07"])
    result = build(gee=gee)["table"]
    late = result[result["system_index"] == "scene1"]
    assert len(late) == len(ROIS) * len(CLOUDS) * 5
    assert (late["turbidity_n"] == 0).all()
    assert late["turbidity_value"].isna().all()


def test_scene_with_no_valid_pixels_is_kept():
    result = build()["table"]
    assert (result["valid_count"] == 0).any()
    assert result["system_index"].nunique() == 2


def test_required_columns_are_present():
    result = build()["table"]
    required = {
        "site_no",
        "system_index",
        "spacecraft",
        "roi",
        "erosion_px",
        "cloud_variant",
        "variant",
        "turbidity_sensor_id",
        "turbidity_approval",
        "turbidity_n",
        "turbidity_dt_min",
        "discharge_estimated_share",
        "stage_estimated_share",
        "cv_group",
        "site_layer",
        "extended_roi_valid",
        "sensitivity_only",
        "qc_level",
    }
    assert required <= set(result.columns)


def test_parameters_follow_site_targets():
    names = table.site_parameters(SITE, CONFIG)
    assert names == {
        "turbidity": "63680",
        "dissolved_oxygen": "00300",
        "discharge": "00060",
        "stage": "00065",
    }
    with_fchl = {**SITE, "targets": {**SITE["targets"], "fchl": True}, "fchl_code": "32320"}
    assert table.site_parameters(with_fchl, CONFIG)["fchl"] == "32320"
    colorado = {**SITE, "targets": {"turbidity": False, "dissolved_oxygen": True, "fchl": False}}
    assert "turbidity" not in table.site_parameters(colorado, CONFIG)


def test_non_target_parameter_columns_are_unmatched_not_zero():
    result = build()["table"]
    assert result["fchl_n"].isna().all()
    assert result["fchl_value"].isna().all()


def test_matched_values_come_from_the_expected_readings():
    result = build()["table"]
    row = result[
        (result["system_index"] == "scene0")
        & (result["variant"] == "nearest")
        & (result["roi"] == "reach")
        & (result["erosion_px"] == 0)
        & (result["cloud_variant"] == "cs50")
    ].iloc[0]
    assert row["turbidity_value"] == 40.0
    assert row["turbidity_dt_min"] == -7.0
    assert row["dissolved_oxygen_value"] == 8.0
    assert row["turbidity_approval"] == "approved"


def test_provisional_values_are_labelled():
    tidy = usgs_frame()
    tidy.loc[tidy["parameter_code"] == "63680", "approval_status"] = "Provisional"
    result = build(tidy=tidy)["table"]
    nearest = result[result["variant"] == "nearest"]
    assert (nearest["turbidity_approval"] == "provisional").all()
    assert (nearest["dissolved_oxygen_approval"] == "approved").all()


def test_estimated_discharge_is_flagged_and_counted_per_site():
    tidy = usgs_frame()
    mask = tidy["parameter_code"] == "00060"
    tidy.loc[mask, "qualifiers"] = "ESTIMATED"
    parts = build(tidy=tidy)
    nearest = parts["table"][parts["table"]["variant"] == "nearest"]
    assert (nearest["discharge_estimated_share"] == 1.0).all()
    assert (nearest["stage_estimated_share"] == 0.0).all()
    estimated = parts["estimated"].set_index("parameter")
    assert estimated.loc["discharge", "estimated_share_readings"] == 1.0
    assert estimated.loc["stage", "estimated_share_readings"] == 0.0
    assert estimated.loc["discharge", "scenes_nearest_estimated"] == 2


def test_extended_roi_flag_is_false_only_for_extended_rows_of_flagged_sites():
    result = build(invalid={"0001": "broken"})["table"]
    flagged = result[~result["extended_roi_valid"]]
    assert set(flagged["roi"]) == {"extended"}
    assert set(flagged["roi_flag_reason"]) == {"broken"}
    assert result[result["roi"] != "extended"]["extended_roi_valid"].all()


def test_unflagged_site_has_valid_extended_roi():
    assert build()["table"]["extended_roi_valid"].all()


def test_layer_group_and_sensitivity_marker():
    colorado = {**SITE, "site_no": "09180500", "layer": "extended_caveat", "cv_group": "colorado"}
    gee = gee_rows("09180500", ["2025-03-01 10:07"])
    tidy = usgs_frame().assign(site_no="09180500")
    result = build(site=colorado, tidy=tidy, gee=gee)["table"]
    assert set(result["site_layer"]) == {"extended_caveat"}
    assert set(result["cv_group"]) == {"colorado"}
    assert result["sensitivity_only"].all()
    assert not build()["table"]["sensitivity_only"].any()


def test_empty_windows_are_counted_per_site_parameter_and_window():
    gee = gee_rows("0001", ["2025-03-01 10:07", "2025-03-20 10:07", "2025-03-21 10:07"])
    counts = build(gee=gee)["empty_windows"]
    turbidity = counts[counts["parameter"] == "turbidity"].set_index("variant")
    assert turbidity.loc["median_15", "scenes"] == 3
    assert turbidity.loc["median_15", "empty_windows"] == 2
    assert bool(turbidity.loc["median_15", "mostly_empty"]) is True
    assert bool(turbidity.loc["daily_mean", "mostly_empty"]) is True


def test_empty_window_counts_ignore_unmatched_parameters():
    counts = build()["empty_windows"]
    assert "fchl" not in set(counts["parameter"])


def test_sixty_minute_series_has_empty_narrow_windows():
    turbidity = regular("2025-03-01 00:00", 72, "60min", code="63680")
    gee = gee_rows("0001", ["2025-03-01 10:25", "2025-03-01 12:35"])
    counts = build(tidy=turbidity, gee=gee)["empty_windows"]
    row = counts[counts["parameter"] == "turbidity"].set_index("variant")
    assert row.loc["median_15", "empty_windows"] == 2
    assert row.loc["median_30", "empty_windows"] == 0
    assert row.loc["nearest", "empty_windows"] == 0


def test_exclusion_counts_per_rule_use_the_requested_level():
    tidy = usgs_frame()
    tidy.loc[(tidy["parameter_code"] == "63680").to_numpy().nonzero()[0][:5], "qualifiers"] = (
        "EQUIP"
    )
    standard = build(tidy=tidy, level="standard")["exclusions"]
    flagged = standard[
        (standard["rule"] == "unreliable_qualifier") & (standard["parameter_code"] == "63680")
    ]
    assert flagged["flagged"].iloc[0] == 5
    assert set(standard["level"]) == {"standard"}


def test_levels_change_the_matched_values_only_through_qc():
    oxygen = regular("2025-03-01 00:00", 96 * 3, "15min", code="00300", values=[8.0] * 288)
    tidy = pd.concat([usgs_frame()[usgs_frame()["parameter_code"] != "00300"], oxygen])
    standard = build(tidy=tidy, level="standard")["table"]
    strict = build(tidy=tidy, level="strict")["table"]
    assert standard["dissolved_oxygen_n"].gt(0).any()
    assert (strict["dissolved_oxygen_n"] == 0).all()


def test_two_sensor_site_keeps_sensor_id_per_row():
    first = regular("2025-03-01 00:00", 96, "15min", code="63680", sensor="old")
    second = regular("2025-03-02 00:00", 96, "15min", code="63680", sensor="new")
    tidy = pd.concat([first, second], ignore_index=True)
    result = build(tidy=tidy)["table"]
    nearest = result[result["variant"] == "nearest"].drop_duplicates("system_index")
    assert nearest.set_index("system_index")["turbidity_sensor_id"].to_dict() == {
        "scene0": "old",
        "scene1": "new",
    }


def test_site_layers_and_roi_flags_parse_config_shapes():
    sites = table.site_layers(
        {"layers": {"core": [{"site_no": "1", "cv_group": "a"}], "extended_caveat": []}}
    )
    assert sites["1"]["layer"] == "core"
    flags = table.invalid_extended_sites(
        {"extended_invalid": {"2": {"reasons": ["broken", "tiny_extended"]}}}
    )
    assert flags == {"2": "broken|tiny_extended"}


def test_unique_scenes_collapses_roi_rows():
    gee = gee_rows("0001", ["2025-03-01 10:07", "2025-03-02 10:07"])
    assert len(table.unique_scenes(gee)) == 2


def write_run(tmp_path):
    repo = tmp_path / "repo"
    usgs = repo / "usgs" / "tidy" / "0001"
    usgs.mkdir(parents=True)
    usgs_frame().to_parquet(usgs / "0001_2025.parquet", index=False)
    gee = repo / "gee" / "run1" / "tidy" / "0001"
    gee.mkdir(parents=True)
    gee_rows("0001", ["2025-03-01 10:07", "2025-03-02 10:07"]).to_parquet(
        gee / "0001_2025.parquet", index=False
    )
    (repo / "sites.yaml").write_text(
        yaml.safe_dump({"layers": {"core": [{**SITE}], "extended_caveat": []}})
    )
    (repo / "flags.yaml").write_text(yaml.safe_dump({"extended_invalid": {}}))
    config = copy.deepcopy(CONFIG)
    config.update(
        {
            "usgs_run_dir": "usgs",
            "gee_root": "gee",
            "sites_file": "sites.yaml",
            "roi_flags_file": "flags.yaml",
            "output_root": "out",
        }
    )
    return repo, config


def test_run_writes_per_level_outputs_and_never_touches_the_gee_folder(tmp_path):
    repo, config = write_run(tmp_path)
    before = sorted(path.name for path in (repo / "gee").rglob("*"))
    out = table.run(config, "run1", ["standard", "strict"], None, repo)
    after = sorted(path.name for path in (repo / "gee").rglob("*"))
    assert before == after
    assert (out / "standard" / "matched_0001.parquet").exists()
    assert (out / "strict" / "scenes_0001.parquet").exists()
    assert (out / "standard" / "empty_windows.csv").exists()
    written = pd.read_parquet(out / "standard" / "matched_0001.parquet")
    assert set(written["qc_level"]) == {"standard"}
    assert set(windows.window_variants(CONFIG)) == set(written["variant"])
    assert np.isfinite(written["turbidity_value"]).any()
