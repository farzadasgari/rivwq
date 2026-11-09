from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import pytest

from rivwq.gee import export, manifest

SITES = ["s1", "s2"]


@pytest.fixture
def harness(monkeypatch, tmp_path):
    config = export.load_config()
    config["workers"] = 1
    calls: list[str] = []
    failing: set[str] = set()
    area_calls: list[int] = []

    def fake_process(config, gauges, site, year, run_dir, run_start, rois=None):
        name = export.job_name(site, year)
        if name in failing:
            raise RuntimeError("boom")
        calls.append(name)
        path = run_dir / "tidy" / site / f"{name}.parquet"
        export.write_parquet(pd.DataFrame({"a": [year, 1]}), path)
        entry = manifest.file_entry(run_dir, path, manifest.utc_now())
        entry["scenes"] = 10
        entry["seconds"] = 1.0
        return entry

    def fake_areas(config, gauges, sites, plan=None):
        area_calls.append(1)
        return pd.DataFrame({"site_no": sites})

    monkeypatch.setattr(export.session, "initialize", lambda: None)
    monkeypatch.setattr(export, "verify_assets", lambda config: {})
    monkeypatch.setattr(export, "process_site_year", fake_process)
    monkeypatch.setattr(export, "compute_roi_areas", fake_areas)

    def run(years, sites=SITES):
        return export.run(config, "job", sites, years, raw_root=tmp_path)

    run_dir = tmp_path / "job"
    return {
        "run": run,
        "calls": calls,
        "failing": failing,
        "areas": area_calls,
        "dir": run_dir,
        "state": lambda: manifest.read_json(run_dir / "run_state.json"),
        "manifest": lambda: manifest.read_json(run_dir / "manifest.json"),
        "config": config,
    }


def test_second_phase_keeps_finished_site_years_and_state(harness):
    harness["run"]([2025])
    before = {item["path"]: item for item in harness["manifest"]()["files"]}
    started = harness["manifest"]()["run_started_utc"]
    harness["calls"].clear()
    harness["run"]([2024])
    assert harness["calls"] == ["s1_2024", "s2_2024"]
    after = {item["path"]: item for item in harness["manifest"]()["files"]}
    for path, item in before.items():
        assert after[path] == item
    state = harness["state"]()
    assert state["done"] == ["s1_2024", "s1_2025", "s2_2024", "s2_2025"]
    assert state["pending"] == []
    assert [phase["years"] for phase in state["phases"]] == [[2025], [2024]]
    assert harness["manifest"]()["run_started_utc"] == started
    assert len(harness["areas"]) == 1


def test_rerunning_a_finished_phase_fetches_nothing(harness):
    harness["run"]([2025])
    harness["calls"].clear()
    harness["run"]([2025])
    assert harness["calls"] == []
    assert len(harness["state"]()["done"]) == 2


def test_a_failure_in_one_phase_survives_the_next_phase(harness):
    harness["failing"].add("s2_2025")
    harness["run"]([2025])
    assert harness["state"]()["failed"] == {"s2_2025": "RuntimeError: boom"}
    harness["run"]([2024])
    assert "s2_2025" in harness["state"]()["failed"]
    harness["failing"].clear()
    harness["run"]([2025])
    state = harness["state"]()
    assert state["failed"] == {}
    assert "s2_2025" in state["done"]


def test_a_damaged_file_from_an_earlier_phase_is_pending_not_silently_done(harness):
    harness["run"]([2025])
    path = harness["dir"] / "tidy" / "s1" / "s1_2025.parquet"
    path.write_bytes(b"damaged")
    harness["calls"].clear()
    harness["run"]([2024])
    assert harness["calls"] == ["s1_2024", "s2_2024"]
    state = harness["state"]()
    assert "s1_2025" in state["pending"]
    assert "s1_2025" not in state["done"]


def test_different_site_list_is_refused_and_leaves_files_alone(harness):
    harness["run"]([2025])
    snapshot = (harness["dir"] / "manifest.json").read_text(encoding="utf-8")
    with pytest.raises(RuntimeError):
        harness["run"]([2024], sites=["s1"])
    assert (harness["dir"] / "manifest.json").read_text(encoding="utf-8") == snapshot


def test_a_live_lock_blocks_a_second_run_and_changes_nothing(harness):
    harness["run"]([2025])
    snapshot = (harness["dir"] / "manifest.json").read_text(encoding="utf-8")
    (harness["dir"] / export.LOCK_NAME).write_text(str(os.getpid()), encoding="utf-8")
    with pytest.raises(RuntimeError):
        harness["run"]([2024])
    assert (harness["dir"] / "manifest.json").read_text(encoding="utf-8") == snapshot
    assert (harness["dir"] / export.LOCK_NAME).exists()


def test_lock_is_released_after_a_normal_run(harness):
    harness["run"]([2025])
    assert not (harness["dir"] / export.LOCK_NAME).exists()


def test_main_passes_the_parsed_years_to_the_run(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        export,
        "run",
        lambda config, run_id, sites, years, skips=None: seen.update(years=years, run_id=run_id),
    )
    export.main(["--mode", "full", "--run-id", "x", "--years", "2025-2026"])
    assert seen == {"years": [2025, 2026], "run_id": "x"}


def test_main_without_years_uses_every_year_up_to_today(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        export, "run", lambda config, run_id, sites, years, skips=None: seen.update(years=years)
    )
    export.main(["--mode", "full", "--run-id", "x"])
    assert seen["years"][0] == 2018
    assert seen["years"][-1] == datetime.now().year


def test_main_rejects_bad_years(monkeypatch):
    monkeypatch.setattr(export, "run", lambda *args: None)
    with pytest.raises(SystemExit):
        export.main(["--mode", "full", "--run-id", "x", "--years", "2017"])
