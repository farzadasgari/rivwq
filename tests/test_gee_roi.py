from __future__ import annotations

import json

import yaml

from rivwq.gee import roi


def gee_config() -> dict:
    return yaml.safe_load((roi.REPO_ROOT / "config" / "gee.yaml").read_text(encoding="utf-8"))


CONFIG = gee_config()


def test_roi_hash_changes_with_variant_and_gauge():
    gauges = {"1": (-95.0, 39.0)}
    base = roi.roi_hash(gauges, CONFIG)
    moved = roi.roi_hash({"1": (-95.1, 39.0)}, CONFIG)
    changed = json.loads(json.dumps(CONFIG))
    changed["roi"]["variants"]["reach"] = 6000
    assert base != moved
    assert base != roi.roi_hash(gauges, changed)
    assert base == roi.roi_hash(gauges, CONFIG)


def test_gauge_file_has_all_frozen_sites():
    gauges = roi.load_gauges(roi.REPO_ROOT / CONFIG["roi"]["gauges_file"])
    frozen = yaml.safe_load((roi.REPO_ROOT / "config" / "sites.yaml").read_text(encoding="utf-8"))
    expected = {item["site_no"] for items in frozen["layers"].values() for item in items}
    assert set(gauges) == expected
    assert len(gauges) == 24
    for longitude, latitude in gauges.values():
        assert -180 < longitude < -60
        assert 20 < latitude < 72


def test_utm_epsg_for_kansas():
    assert roi.utm_epsg(-95.68) == "EPSG:32615"


def test_hash_is_unchanged_without_skips_and_changes_with_them():
    gauges = {"1": (-95.0, 39.0), "2": (-96.0, 40.0)}
    base = roi.roi_hash(gauges, CONFIG)
    assert roi.roi_hash(gauges, CONFIG, {}) == base
    assert roi.roi_hash(gauges, CONFIG, None) == base
    skipped = roi.roi_hash(gauges, CONFIG, {"1": ["extended"]})
    assert skipped != base
    assert roi.roi_hash(gauges, CONFIG, {"2": ["extended"]}) != skipped
    assert roi.roi_hash(gauges, CONFIG, {"1": ["reach"]}) != skipped


def test_hash_does_not_depend_on_the_order_of_skips():
    gauges = {"1": (-95.0, 39.0), "2": (-96.0, 40.0)}
    first = roi.roi_hash(gauges, CONFIG, {"1": ["extended"], "2": ["extended"]})
    second = roi.roi_hash(gauges, CONFIG, {"2": ["extended"], "1": ["extended"]})
    assert first == second
