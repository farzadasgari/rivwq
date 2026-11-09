from __future__ import annotations

import pytest
import yaml

from rivwq.gee import extract, roi, session, stats

WAMEGO = "06887500"


@pytest.mark.gee
def test_one_scene_has_every_expected_column():
    import ee

    session.initialize()
    config = yaml.safe_load((roi.REPO_ROOT / "config" / "gee.yaml").read_text(encoding="utf-8"))
    gauges = roi.load_gauges(roi.REPO_ROOT / config["roi"]["gauges_file"])
    point = ee.Geometry.Point(list(gauges[WAMEGO]))
    start, end = "2023-06-01T00:00:00Z", "2023-06-10T00:00:00Z"
    ids = extract.scene_ids(config, point, start, end)
    polygons = extract.water_polygons(config, point)
    geometries = {name: polygon.getInfo() for name, polygon in polygons.items()}
    collection = extract.scene_batch_collection(config, point, start, end, ids[:1], geometries)
    properties = collection.getInfo()["features"][0]["properties"]
    assert stats.total_name("reach", 1) in properties
    assert stats.column_name("reach", 1, "cs60", "B8A", "median") in properties
    assert stats.count_name("extended", 0, "scl") in properties
