from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from rivwq.gee import roi, stats

FIXTURE = Path(__file__).parent / "fixtures" / "gee_twin.json"


def gee_config() -> dict:
    return yaml.safe_load((roi.REPO_ROOT / "config" / "gee.yaml").read_text(encoding="utf-8"))


CONFIG = gee_config()


def fixture_arrays() -> dict:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {
        "raw": raw,
        "water": np.array(raw["water"], dtype=bool),
        "footprint": np.array(raw["footprint"], dtype=bool),
        "score": np.array(raw["cloud_score"]),
        "classes": np.array(raw["classification"]),
        "bands": {name: np.array(values, dtype=float) for name, values in raw["bands"].items()},
    }


def test_twin_matches_frozen_fixture():
    data = fixture_arrays()
    result = stats.twin_scene_statistics(
        data["bands"], data["score"], data["classes"], data["water"], data["footprint"], CONFIG
    )
    for key, expected in data["raw"]["expected"].items():
        erosion, tag = key.split("|")
        entry = result[(int(erosion), tag)]
        assert entry["total_count"] == expected["total_count"]
        assert entry["valid_count"] == expected["valid_count"]
        observed = entry["B2"]
        for name in ("mean", "median", "std"):
            if expected[name] is None:
                assert np.isnan(observed[name])
            else:
                assert observed[name] == pytest.approx(expected[name])


def test_erode_plus_removes_border_of_full_block():
    block = np.ones((5, 5), dtype=bool)
    eroded = stats.erode_plus(block)
    assert eroded.sum() == 9
    assert not eroded[0].any()


def test_cloud_tags_follow_config():
    assert stats.cloud_tags(CONFIG) == ["cs50", "cs60", "cs70", "scl"]


def test_footprint_excludes_pixels_from_total():
    data = fixture_arrays()
    footprint = data["footprint"].copy()
    footprint[0, 0] = False
    result = stats.twin_scene_statistics(
        data["bands"], data["score"], data["classes"], data["water"], footprint, CONFIG
    )
    assert result[(0, "cs50")]["total_count"] == 8


def test_unknown_spacecraft_fails_loudly():
    with pytest.raises(KeyError):
        stats.spacecraft_code("Sentinel-2D", CONFIG["spacecraft"])


def feature_for(config: dict, name: str = "Sentinel-2C", rois: list[str] | None = None) -> dict:
    properties = {
        "system_index": "20250101T000000_20250101T000000_T14SQJ",
        "MGRS_TILE": "14SQJ",
        "SENSING_ORBIT_NUMBER": 112,
        "SPACECRAFT_NAME": name,
        "PROCESSING_BASELINE": "05.11",
        "MEAN_SOLAR_ZENITH_ANGLE": 40.0,
        "MEAN_SOLAR_AZIMUTH_ANGLE": 150.0,
        "MEAN_INCIDENCE_ZENITH_ANGLE_B8A": 3.0,
        "MEAN_INCIDENCE_AZIMUTH_ANGLE_B8A": 190.0,
        "CLOUDY_PIXEL_PERCENTAGE": 10.0,
        "system:time_start": 1735689600000,
        "has_cloud_score": True,
    }
    for roi_name in rois or list(config["roi"]["variants"]):
        for erosion in config["masks"]["erosion_pixels"]:
            properties[stats.total_name(roi_name, erosion)] = 100
            for tag in stats.cloud_tags(config):
                properties[stats.count_name(roi_name, erosion, tag)] = 80
                for band in stats.band_list(config):
                    for suffix in stats.REDUCER_SUFFIX.values():
                        key = stats.column_name(roi_name, erosion, tag, band, suffix)
                        properties[key] = 1.0
    return {"properties": properties}


def test_long_table_shape_and_columns():
    wide = stats.scene_table([feature_for(CONFIG)], CONFIG)
    long = stats.to_long(wide, "123", CONFIG)
    assert len(long) == 3 * 2 * 4
    assert list(long.columns) == stats.long_columns(CONFIG)
    assert set(long["spacecraft"]) == {"S2C"}
    assert long["valid_fraction"].iloc[0] == pytest.approx(0.8)
    assert long["time_utc"].iloc[0] == pd.Timestamp("2025-01-01", tz="UTC")


def test_long_table_only_has_the_requested_rois():
    kept = ["near_gauge", "reach"]
    wide = stats.scene_table([feature_for(CONFIG, rois=kept)], CONFIG)
    long = stats.to_long(wide, "123", CONFIG, kept)
    assert len(long) == 2 * 2 * 4
    assert set(long["roi"]) == set(kept)
    assert "extended" not in set(long["roi"])
    assert list(long.columns) == stats.long_columns(CONFIG)


def test_asking_for_a_roi_that_was_not_extracted_fails_loudly():
    kept = ["near_gauge", "reach"]
    wide = stats.scene_table([feature_for(CONFIG, rois=kept)], CONFIG)
    with pytest.raises(KeyError):
        stats.to_long(wide, "123", CONFIG)
