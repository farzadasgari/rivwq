from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from rivwq.gee import export, roi, roi_plan, stats

FLAGGED = "06887500"
VALID = "03378500"
SITES = [FLAGGED, VALID]
RUN_START = datetime(2026, 11, 8, tzinfo=UTC)


class Reply:
    def __init__(self, value):
        self.value = value

    def getInfo(self):
        return self.value


def feature(config: dict, rois: list[str], index: str) -> dict:
    properties = {
        "system_index": index,
        "MGRS_TILE": "14SQJ",
        "SENSING_ORBIT_NUMBER": 1,
        "SPACECRAFT_NAME": "Sentinel-2B",
        "PROCESSING_BASELINE": "05.11",
        "MEAN_SOLAR_ZENITH_ANGLE": 1.0,
        "MEAN_SOLAR_AZIMUTH_ANGLE": 1.0,
        "MEAN_INCIDENCE_ZENITH_ANGLE_B8A": 1.0,
        "MEAN_INCIDENCE_AZIMUTH_ANGLE_B8A": 1.0,
        "CLOUDY_PIXEL_PERCENTAGE": 1.0,
        "system:time_start": 1735689600000,
        "has_cloud_score": 1,
    }
    for name in rois:
        for erosion in config["masks"]["erosion_pixels"]:
            properties[stats.total_name(name, erosion)] = 10
            for tag in stats.cloud_tags(config):
                properties[stats.count_name(name, erosion, tag)] = 5
                for band in stats.band_list(config):
                    for suffix in stats.REDUCER_SUFFIX.values():
                        properties[stats.column_name(name, erosion, tag, band, suffix)] = 1.0
    return {"properties": properties}


@pytest.fixture
def env(monkeypatch, tmp_path):
    config = export.load_config()
    config["workers"] = 1
    asked = {"polygons": [], "geometries": [], "areas": []}

    def water_polygons(config, point, rois=None):
        asked["polygons"].append(list(rois))
        return {name: Reply({"type": "Polygon", "name": name}) for name in rois}

    def scene_ids(config, point, start, end):
        return ["a", "b"]

    def scene_batch_collection(config, point, start, end, ids, geometries):
        asked["geometries"].append(sorted(geometries))
        features = [feature(config, sorted(geometries), name) for name in ids]
        return Reply({"features": features})

    def roi_area_table(config, point, epsg, rois=None):
        asked["areas"].append(list(rois))
        values = {f"{name}__e{e}": 1.0 for name in rois for e in config["masks"]["erosion_pixels"]}
        return Reply(values)

    monkeypatch.setattr(export.session, "initialize", lambda: None)
    monkeypatch.setattr(export, "verify_assets", lambda config: {})
    monkeypatch.setattr(export, "point_for", lambda gauges, site: None)
    monkeypatch.setattr(export.extract, "water_polygons", water_polygons)
    monkeypatch.setattr(export.extract, "scene_ids", scene_ids)
    monkeypatch.setattr(export.extract, "scene_batch_collection", scene_batch_collection)
    monkeypatch.setattr(export.extract, "roi_area_table", roi_area_table)
    gauges = roi.load_gauges(roi.REPO_ROOT / config["roi"]["gauges_file"])
    skips = roi_plan.skips_from_flags(config, sorted(gauges))
    return {
        "config": config,
        "asked": asked,
        "dir": tmp_path / "job",
        "skips": skips,
        "root": tmp_path,
        "gauges": gauges,
    }


def run_job(env, skips, years=(2025,), run_id="job"):
    return export.run(env["config"], run_id, SITES, list(years), raw_root=env["root"], skips=skips)


def tidy(env, site: str, year: int = 2025) -> pd.DataFrame:
    return pd.read_parquet(env["dir"] / "tidy" / site / f"{site}_{year}.parquet")


def test_the_flagged_site_never_asks_for_the_skipped_roi(env):
    run_job(env, env["skips"])
    assert ["near_gauge", "reach"] in env["asked"]["polygons"]
    assert ["near_gauge", "reach", "extended"] in env["asked"]["polygons"]
    assert ["extended"] not in env["asked"]["polygons"]
    assert ["near_gauge", "reach"] in env["asked"]["geometries"]
    assert ["near_gauge", "reach"] in env["asked"]["areas"]
    flagged_requests = [
        entry
        for entry in env["asked"]["polygons"] + env["asked"]["areas"]
        if "extended" not in entry
    ]
    assert len(flagged_requests) == 2


def test_the_valid_site_still_gets_every_roi(env):
    run_job(env, env["skips"])
    assert set(tidy(env, VALID)["roi"]) == {"near_gauge", "reach", "extended"}
    assert set(tidy(env, FLAGGED)["roi"]) == {"near_gauge", "reach"}
    assert len(tidy(env, VALID)) == 2 * 3 * 2 * 4
    assert len(tidy(env, FLAGGED)) == 2 * 2 * 2 * 4


def test_no_extended_column_values_are_stored_for_the_flagged_site(env):
    run_job(env, env["skips"])
    areas = pd.read_parquet(env["dir"] / "roi_areas.parquet")
    flagged = areas[areas["site_no"] == FLAGGED]
    assert set(flagged["roi"]) == {"near_gauge", "reach"}
    assert set(areas[areas["site_no"] == VALID]["roi"]) == {"near_gauge", "reach", "extended"}
    geometries = pd.read_json(env["dir"] / "roi_geometries" / f"{FLAGGED}.json", typ="series")
    assert sorted(geometries.index) == ["near_gauge", "reach"]


def test_manifest_records_the_plan_the_skips_and_the_hash(env):
    run_job(env, env["skips"])
    current = export.manifest.read_json(env["dir"] / "manifest.json")
    assert current["roi_plan"][FLAGGED] == ["near_gauge", "reach"]
    assert current["roi_plan"][VALID] == ["near_gauge", "reach", "extended"]
    assert current["skipped_site_rois"] == [f"{FLAGGED}:extended"]
    assert current["roi_hash"] == roi.roi_hash(
        env["gauges"], env["config"], {FLAGGED: ["extended"]}
    )
    assert current["roi_flags_sha256"] == roi_plan.flags_sha256(env["config"])
    assert sorted(current["sites"]) == sorted(SITES)


def test_a_run_without_skips_keeps_the_old_hash_and_every_roi(env):
    run_job(env, None)
    current = export.manifest.read_json(env["dir"] / "manifest.json")
    assert current["roi_hash"] == roi.roi_hash(env["gauges"], env["config"])
    assert current["skipped_site_rois"] == []
    assert set(tidy(env, FLAGGED)["roi"]) == {"near_gauge", "reach", "extended"}


def test_resuming_with_a_different_plan_is_refused_and_changes_nothing(env):
    run_job(env, env["skips"])
    before = (env["dir"] / "manifest.json").read_text(encoding="utf-8")
    with pytest.raises(RuntimeError):
        run_job(env, None, years=(2024,))
    assert (env["dir"] / "manifest.json").read_text(encoding="utf-8") == before


def test_a_later_phase_with_the_same_plan_resumes(env):
    run_job(env, env["skips"], years=(2025,))
    run_job(env, env["skips"], years=(2024,))
    state = export.manifest.read_json(env["dir"] / "run_state.json")
    assert len(state["done"]) == 4
    assert set(tidy(env, FLAGGED, 2024)["roi"]) == {"near_gauge", "reach"}


def test_stored_geometries_that_do_not_match_the_plan_are_refused(env, tmp_path):
    folder = tmp_path / "geo"
    export.manifest.atomic_write_json(
        folder / "roi_geometries" / "s.json", {"near_gauge": {}, "reach": {}, "extended": {}}
    )
    with pytest.raises(RuntimeError):
        export.load_geometries(env["config"], {}, "s", folder, ["near_gauge", "reach"])


def test_full_mode_reads_the_skips_from_the_flags_file_and_pilot_uses_none(monkeypatch):
    seen = {}

    def fake_run(config, run_id, sites, years, raw_root=None, skips=None):
        seen[run_id] = skips

    monkeypatch.setattr(export, "run", fake_run)
    export.main(["--mode", "full", "--run-id", "full_run", "--years", "2025"])
    export.main(["--mode", "pilot", "--run-id", "pilot_run"])
    assert len(seen["full_run"]) == 10
    assert all(names == ["extended"] for names in seen["full_run"].values())
    assert seen["pilot_run"] is None


def test_skips_for_sites_outside_the_run_are_ignored(env):
    export.run(env["config"], "job", [VALID], [2025], raw_root=env["root"], skips=env["skips"])
    current = export.manifest.read_json(env["dir"] / "manifest.json")
    assert current["skipped_site_rois"] == []
    assert current["roi_hash"] == roi.roi_hash(env["gauges"], env["config"])
