from __future__ import annotations

import pytest

from rivwq.gee import estimate

CONFIG = {
    "date_start": "2018-01-01",
    "workers": 2,
    "phases": [{"name": "p1", "years": "2025-2026"}, {"name": "p2", "years": "2024"}],
    "roi": {"variants": {"near_gauge": 1000, "reach": 5000, "extended": 20000}},
    "roi_plan": {"skipped_roi": "extended"},
    "eta": {
        "seconds_per_scene": 10.0,
        "scenes_per_date": 2.0,
        "scenes_per_site_year": 100,
        "skipped_roi_cost_factor": 0.5,
        "scene_factor_by_year": {2025: 1.5},
    },
}
DATES = {"a": 100.0, "b": 50.0}
FULL = {"a": ["near_gauge", "reach", "extended"], "b": ["near_gauge", "reach", "extended"]}
SKIPPED = {"a": ["near_gauge", "reach"], "b": ["near_gauge", "reach", "extended"]}


def test_scenes_scale_with_dates_and_year_factor():
    assert estimate.scenes_for("a", 2024, DATES, CONFIG["eta"]) == 200.0
    assert estimate.scenes_for("a", 2025, DATES, CONFIG["eta"]) == 300.0
    assert estimate.scenes_for("b", 2024, DATES, CONFIG["eta"]) == 100.0


def test_phase_hours_without_skips():
    rows = {row["name"]: row for row in estimate.phase_hours(CONFIG, FULL, DATES, 2026)}
    assert rows["p2"]["scenes"] == 300.0
    assert rows["p2"]["hours"] == pytest.approx(300 * 10.0 / 2 / 3600)
    assert rows["p1"]["years"] == [2025, 2026]


def test_skipping_the_roi_lowers_only_the_flagged_sites_hours():
    full = {row["name"]: row for row in estimate.phase_hours(CONFIG, FULL, DATES, 2026)}
    cut = {row["name"]: row for row in estimate.phase_hours(CONFIG, SKIPPED, DATES, 2026)}
    assert cut["p2"]["scenes"] == full["p2"]["scenes"]
    expected = (200 * 0.5 + 100) * 10.0 / 2 / 3600
    assert cut["p2"]["hours"] == pytest.approx(expected)
    assert cut["p2"]["hours"] < full["p2"]["hours"]


def test_dates_table_is_read_by_site(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("site,dates_per_year\n01184000,100.5\n", encoding="utf-8")
    assert estimate.load_dates_per_year(path) == {"01184000": 100.5}
