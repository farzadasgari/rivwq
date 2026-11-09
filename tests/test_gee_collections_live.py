from __future__ import annotations

import pytest
import yaml

from rivwq.gee import collections, roi, session

WAMEGO = "06887500"


@pytest.mark.gee
def test_joined_scenes_carry_cloud_score_flag():
    import ee

    session.initialize()
    config = yaml.safe_load((roi.REPO_ROOT / "config" / "gee.yaml").read_text(encoding="utf-8"))
    gauges = roi.load_gauges(roi.REPO_ROOT / config["roi"]["gauges_file"])
    point = ee.Geometry.Point(list(gauges[WAMEGO]))
    scenes = collections.joined_scenes(config, point, "2023-06-01", "2023-06-15")
    first = scenes.first()
    assert scenes.size().getInfo() > 0
    assert first.get("has_cloud_score").getInfo() == 1
    assert "cs_cdf" in first.bandNames().getInfo()
