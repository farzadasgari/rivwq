import hashlib

import pandas as pd
import pytest
import yaml

from rivwq.analysis import sites

SETTINGS = {
    "separators": ["AT", "NR", "NEAR", "AB", "BL", "ABOVE", "BELOW"],
    "punctuation": [","],
    "suffixes": ["RIVER", "RV", "R"],
    "aliases": {"big_blue": "kansas"},
}
FULL = {"river_group": SETTINGS, "basin_digits": 4}


def attributes(huc8: dict[str, str | None], fchl: str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"huc8": pd.Series(huc8), "chlorophyll_fluorescence_code": fchl}, dtype=object
    )
    frame.index.name = "site_no"
    return frame


def row(**overrides) -> dict:
    base = {
        "station": "TEST RIVER AT PLACE, KS",
        "state": "KS",
        "region": "Midwest",
        "role": "pool",
        "drainage_area_sqmi": 2000.0,
        "pixels_across": 12.0,
        "pass_width_10px": True,
        "pass_turbidity_cov": True,
        "pass_overlap": True,
        "pass_overlap_conservative": True,
        "caveat": False,
        "caveat_reason": None,
        "coverage_estimated": False,
        "target_turbidity_ok": True,
        "target_do_ok": True,
        "target_fchl_ok": False,
        "turbidity_class": "turbid",
        "regulation_class": "regulated",
        "regime_class": "rain_fed",
        "cloud_class": "sunnier",
        "est_scenes_upper": 100.0,
        "est_scenes_matched": 80.0,
    }
    return {**base, **overrides}


def table(rows: dict[str, dict]) -> pd.DataFrame:
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "site_no"
    return frame


def test_layers_follow_the_stated_rules():
    frame = table(
        {
            "core": row(),
            "caveated": row(caveat=True, caveat_reason="head of tide"),
            "estimated": row(coverage_estimated=True, pass_overlap_conservative=False),
            "narrow": row(pass_width_10px=False),
            "low_coverage": row(pass_turbidity_cov=False),
            "no_overlap": row(pass_overlap_conservative=False, pass_overlap=False),
            "estimated_no_overlap": row(coverage_estimated=True, pass_overlap=False),
        }
    )
    layers = sites.assign_layers(frame)
    assert list(layers["core"].index) == ["core"]
    assert list(layers["extended_caveat"].index) == ["caveated"]
    assert list(layers["extended_estimated"].index) == ["estimated"]


OVERRIDE = {"low_coverage": {"layer": "extended_caveat", "caveat_reason": "fails 2025 rule"}}


def test_override_moves_a_site_that_fails_the_rules_into_extended_caveat():
    frame = table({"core": row(), "low_coverage": row(pass_turbidity_cov=False)})
    layers = sites.assign_layers(sites.apply_overrides(frame, OVERRIDE), OVERRIDE)
    assert list(layers["core"].index) == ["core"]
    assert list(layers["extended_caveat"].index) == ["low_coverage"]


def test_override_takes_a_passing_site_out_of_core():
    frame = table({"a": row(), "b": row()})
    overrides = {"b": OVERRIDE["low_coverage"]}
    layers = sites.assign_layers(sites.apply_overrides(frame, overrides), overrides)
    assert list(layers["core"].index) == ["a"]
    assert list(layers["extended_caveat"].index) == ["b"]


def test_override_sets_caveat_and_reason_on_the_record():
    frame = table({"low_coverage": row(pass_turbidity_cov=False)})
    layers = sites.build_sites(
        frame,
        attributes({"low_coverage": "10270104"}),
        {**FULL, "layer_overrides": OVERRIDE},
    )
    record = layers["extended_caveat"][0]
    assert record["caveat"] is True
    assert record["caveat_reason"] == "fails 2025 rule"


def test_override_for_unknown_site_or_layer_is_rejected():
    frame = table({"a": row()})
    with pytest.raises(sites.FreezeError):
        sites.apply_overrides(frame, {"zzz": OVERRIDE["low_coverage"]})
    with pytest.raises(sites.FreezeError):
        sites.apply_overrides(frame, {"a": {"layer": "core", "caveat_reason": "x"}})


def test_without_overrides_the_layers_are_unchanged():
    frame = table({"a": row(), "b": row(pass_turbidity_cov=False)})
    layers = sites.build_sites(frame, attributes({"a": "10270104"}), FULL)
    assert [record["site_no"] for record in layers["core"]] == ["a"]
    assert layers["extended_caveat"] == []


def test_caveat_site_that_is_also_estimated_is_rejected():
    frame = table({"both": row(caveat=True, coverage_estimated=True)})
    with pytest.raises(sites.FreezeError):
        sites.assign_layers(frame)


@pytest.mark.parametrize(
    ("station", "expected"),
    [
        ("KANSAS R AB TOPEKA WEIR AT TOPEKA, KS", "kansas"),
        ("KANSAS R AT WAMEGO, KS", "kansas"),
        ("Delaware River at Trenton NJ", "delaware"),
        ("POTOMAC RIVER NEAR WASH, DC LITTLE FALLS PUMP STA", "potomac"),
        ("W Fk San Jacinto Rv nr Humble, TX", "w_fk_san_jacinto"),
        ("OHIO RIVER AT IRONTON, OH", "ohio"),
        ("BIG BLUE R NR MANHATTAN, KS", "kansas"),
    ],
)
def test_river_group_from_station_name(station, expected):
    assert sites.river_group(station, SETTINGS) == expected


def test_site_numbers_survive_a_yaml_round_trip_as_text():
    frame = table({"01438500": row()})
    layers = sites.build_sites(frame, attributes({"01438500": "02040101"}, "32316"), FULL)
    document = sites.freeze_document(layers, "t.csv", "abc", 0.25)
    loaded = yaml.safe_load(sites.render(document))
    record = loaded["layers"]["core"][0]
    assert record["site_no"] == "01438500"
    assert record["fchl_code"] == "32316"
    assert record["targets"] == {"turbidity": True, "dissolved_oxygen": True, "fchl": False}
    assert record["caveat_reason"] is None
    assert record["basin_group"] == "0204"


def test_caveat_reason_is_kept_for_caveated_sites():
    frame = table({"a": row(caveat=True, caveat_reason="head of tide")})
    layers = sites.build_sites(frame, attributes({"a": "10270104", "b": "10270104"}), FULL)
    record = layers["extended_caveat"][0]
    assert record["caveat"] is True
    assert record["caveat_reason"] == "head of tide"
    assert record["fchl_code"] is None


def test_scene_totals_are_summed_per_layer():
    frame = table({"a": row(), "b": row(est_scenes_upper=50.0, est_scenes_matched=40.0)})
    layers = sites.build_sites(frame, attributes({"a": "10270104", "b": "10270104"}), FULL)
    assert sites.scene_totals(layers)["core"] == {
        "est_scenes_upper": 150.0,
        "est_scenes_matched": 120.0,
    }
    assert sites.scene_totals(layers)["extended_caveat"]["est_scenes_matched"] == 0


def test_count_check_passes_and_fails_loudly():
    settings = {"expected_counts": {"core": 1}, "frozen_table_sha256": "same"}
    sites.check_counts({"core": 1}, settings, "same")
    with pytest.raises(sites.FreezeError, match="unchanged"):
        sites.check_counts({"core": 2}, settings, "same")
    with pytest.raises(sites.FreezeError, match="table changed"):
        sites.check_counts({"core": 2}, settings, "other")


def test_read_table_rejects_missing_rule_values(tmp_path):
    frame = table({"a": row()}).reset_index().rename(columns={"site_no": "site"})
    frame["caveat"] = frame["caveat"].astype(object)
    frame.loc[0, "caveat"] = None
    path = tmp_path / "table.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(sites.FreezeError):
        sites.read_table(path)


def test_file_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "x.txt"
    path.write_bytes(b"abc")
    assert sites.file_sha256(path) == hashlib.sha256(b"abc").hexdigest()


def test_basin_group_is_first_digits_of_huc8():
    assert sites.basin_group("10270104", 4) == "1027"


def test_missing_huc8_aborts():
    frame = table({"a": row()})
    with pytest.raises(sites.FreezeError, match="HUC8"):
        sites.build_sites(frame, attributes({"b": "10270104"}), FULL)
    with pytest.raises(sites.FreezeError, match="HUC8"):
        sites.build_sites(frame, attributes({"a": None}), FULL)


def test_grouping_conflicts_are_flagged_both_ways():
    frame = table(
        {
            "a": row(station="KANSAS R AT X, KS"),
            "b": row(station="KANSAS R AT Y, KS"),
            "c": row(station="OTHER R AT Z, KS"),
            "d": row(station="THIRD R AT W, KS"),
        }
    )
    huc = {"a": "10270104", "b": "10300101", "c": "10270199", "d": "20000000"}
    layers = sites.build_sites(frame, attributes(huc), FULL)
    found = sites.grouping_conflicts(layers)
    assert found["same_river_different_basin"] == [("a", "b")]
    assert found["same_basin_different_river"] == [("a", "c")]


def test_attributes_fall_back_to_later_files(tmp_path):
    header = "site_no,huc8,chlorophyll_fluorescence_code\n"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text(header + "01,11111111,32315\n")
    second.write_text(header + "01,99999999,32320\n02,22222222,\n")
    combined = sites.read_site_attributes([first, second])
    assert combined.loc["01", "huc8"] == "11111111"
    assert combined.loc["02", "huc8"] == "22222222"


def cv_layers(rivers_and_hucs: dict[str, tuple[str, str]], **fields) -> dict[str, list[dict]]:
    frame = table(
        {
            site: row(station=f"{river} R AT X, KS", **fields)
            for site, (river, _) in rivers_and_hucs.items()
        }
    )
    huc = {site: huc8 for site, (_, huc8) in rivers_and_hucs.items()}
    return sites.build_sites(frame, attributes(huc), FULL)


def test_chain_through_river_then_basin_gives_one_cv_group():
    layers = cv_layers(
        {"A": ("ALPHA", "11110001"), "B": ("ALPHA", "22220001"), "C": ("GAMMA", "22220002")}
    )
    groups = {site["cv_group"] for site in layers["core"]}
    assert groups == {"alpha+gamma"}


def test_unrelated_sites_stay_separate_cv_groups():
    layers = cv_layers({"A": ("ALPHA", "11110001"), "B": ("BETA", "22220001")})
    assert {site["site_no"]: site["cv_group"] for site in layers["core"]} == {
        "A": "alpha",
        "B": "beta",
    }


def test_basin_link_alone_merges_groups():
    layers = cv_layers({"A": ("ALPHA", "11110001"), "B": ("BETA", "11110099")})
    assert {site["cv_group"] for site in layers["core"]} == {"alpha+beta"}


def test_cv_group_summary_flags_large_core_share():
    layers = cv_layers(
        {"A": ("ALPHA", "11110001"), "B": ("ALPHA", "22220001"), "C": ("BETA", "33330001")}
    )
    layers["core"][2]["est_scenes_matched"] = 40.0
    summary = sites.cv_group_summary(layers, 0.25)
    assert summary["alpha"]["members"] == ["A", "B"]
    assert summary["alpha"]["sites_per_layer"]["core"] == 2
    assert summary["alpha"]["core_scene_share"] == pytest.approx(160 / 200)
    assert summary["alpha"]["exceeds_core_scene_share_limit"] is True
    assert summary["beta"]["exceeds_core_scene_share_limit"] is False
    assert summary["beta"]["core_scene_share"] == pytest.approx(40 / 200)
    assert summary["alpha"]["est_scenes_upper"] == 200.0


def test_cv_groups_cover_every_site_once():
    layers = cv_layers({"A": ("ALPHA", "11110001"), "B": ("BETA", "22220001")})
    summary = sites.cv_group_summary(layers, 0.25)
    members = [site for entry in summary.values() for site in entry["members"]]
    assert sorted(members) == ["A", "B"]
