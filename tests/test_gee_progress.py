from __future__ import annotations

import pytest
import yaml

from rivwq.gee import progress, roi

CONFIG = {
    "date_start": "2018-01-01",
    "workers": 2,
    "phases": [{"name": "p1", "years": "2025-2026"}, {"name": "p2", "years": "2024"}],
    "eta": {
        "seconds_per_scene": 10.0,
        "scenes_per_site_year": 100,
        "scene_factor_by_year": {2025: 2.0},
    },
}


def entry(site: str, year: int, scenes: int, seconds: float) -> dict:
    return {
        "path": f"tidy/{site}/{site}_{year}.parquet",
        "scenes": scenes,
        "seconds": seconds,
        "rows": 1,
    }


def sample() -> tuple[dict, dict]:
    manifest = {
        "total_rows": 2,
        "sites": ["a", "b"],
        "files": [entry("a", 2024, 100, 500.0), {"path": "roi_areas.parquet", "rows": 1}],
    }
    state = {
        "run_id": "r",
        "updated_utc": "t",
        "done": ["a_2024"],
        "pending": ["b_2024", "a_2025", "b_2025"],
        "failed": {},
    }
    return state, manifest


def test_seconds_per_scene_comes_from_finished_files_or_the_default():
    _, manifest = sample()
    assert progress.seconds_per_scene(manifest, CONFIG["eta"]) == 5.0
    assert progress.seconds_per_scene({"files": []}, CONFIG["eta"]) == 10.0


def test_expected_scenes_use_site_history_and_year_factor():
    _, manifest = sample()
    per_site = progress.scenes_by_site(manifest, CONFIG["eta"])
    assert per_site == {"a": 100.0}
    assert progress.expected_scenes("a", 2025, per_site, CONFIG["eta"]) == 200.0
    assert progress.expected_scenes("b", 2024, per_site, CONFIG["eta"]) == 100.0


def test_remaining_hours_use_workers_and_observed_rate():
    _, manifest = sample()
    hours = progress.remaining_hours(["b_2024"], manifest, CONFIG)
    assert hours == pytest.approx(100 * 5.0 / 2 / 3600)


def test_report_has_progress_per_year_and_per_phase():
    state, manifest = sample()
    result = progress.report(state, manifest, CONFIG, False)
    assert result["by_year"][2024]["done"] == 1
    assert result["by_year"][2024]["total"] == 2
    assert result["by_year"][2025]["done"] == 0
    assert result["by_year"][2025]["pending"] == 2
    phases = {phase["name"]: phase for phase in result["by_phase"]}
    assert phases["p1"]["years"] == [2025, 2026]
    assert phases["p1"]["total"] == 4
    assert phases["p2"]["done"] == 1
    assert result["overall"]["total"] == 6
    assert result["overall"]["done"] == 1


def test_failed_site_years_are_counted_apart_from_pending():
    state, manifest = sample()
    state["pending"].remove("b_2025")
    state["failed"] = {"b_2025": "boom"}
    result = progress.report(state, manifest, CONFIG, False)
    assert result["by_year"][2025]["failed"] == 1
    assert result["by_year"][2025]["pending"] == 1


def test_selected_years_restrict_the_report():
    state, manifest = sample()
    result = progress.report(state, manifest, CONFIG, False, [2025, 2026])
    assert sorted(result["by_year"]) == [2025, 2026]
    assert [phase["name"] for phase in result["by_phase"]] == ["p1"]
    assert result["overall"]["total"] == 4


def test_finished_run_has_zero_hours_left():
    state, manifest = sample()
    state["done"] = state["done"] + state["pending"]
    state["pending"] = []
    manifest["sites"] = ["a"]
    state["done"] = ["a_2024", "a_2025", "a_2026"]
    result = progress.report(state, manifest, CONFIG, False)
    assert result["overall"]["eta_hours"] == 0.0


def test_text_report_names_phases_and_years():
    state, manifest = sample()
    text = progress.format_report(progress.report(state, manifest, CONFIG, True))
    assert "p1 2025-2026" in text
    assert "p2 2024:" in text
    assert "2024: 1/2 done" in text
    assert "process_alive=True" in text


def test_real_phase_list_covers_every_year_once():
    config = yaml.safe_load((roi.REPO_ROOT / "config" / "gee.yaml").read_text(encoding="utf-8"))
    covered = []
    for phase in config["phases"]:
        covered += progress.phase_years(phase, 2018, 2026)
    assert sorted(covered) == list(range(2018, 2027))
    assert len(covered) == len(set(covered))
