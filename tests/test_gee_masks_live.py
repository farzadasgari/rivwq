from __future__ import annotations

import pytest
import yaml

from rivwq.gee import masks, roi, session

WAMEGO = "06887500"


def load_setup():
    import ee

    config = yaml.safe_load((roi.REPO_ROOT / "config" / "gee.yaml").read_text(encoding="utf-8"))
    gauges = roi.load_gauges(roi.REPO_ROOT / config["roi"]["gauges_file"])
    return config, ee.Geometry.Point(list(gauges[WAMEGO]))


@pytest.mark.gee
def test_connected_water_reaches_beyond_the_seed():
    import ee

    session.initialize()
    config, point = load_setup()
    water = masks.connected_water(config, point)
    total = water.reduceRegion(
        ee.Reducer.sum(), point.buffer(5000), 30, maxPixels=1e9, crs=water.projection()
    ).get("water")
    assert total.getInfo() > 1000
