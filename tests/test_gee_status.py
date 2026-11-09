from __future__ import annotations

import os

from rivwq.gee import status


def sample_state() -> dict:
    return {
        "run_id": "r",
        "updated_utc": "2026-11-09T00:00:00Z",
        "done": ["a_2023", "b_2023"],
        "pending": ["c_2023", "d_2023", "e_2023", "f_2023"],
        "failed": {"g_2023": "boom"},
    }


def sample_manifest() -> dict:
    return {
        "total_rows": 100,
        "files": [
            {"path": "tidy/a/a_2023.parquet", "seconds": 1000.0, "scenes": 10},
            {"path": "tidy/b/b_2023.parquet", "seconds": 2000.0, "scenes": 20},
            {"path": "roi_areas.parquet"},
        ],
    }


def test_summary_counts_and_eta():
    summary = status.summarise(sample_state(), sample_manifest(), 2, True)
    assert summary["done"] == 2
    assert summary["pending"] == 4
    assert summary["failed"] == 1
    assert summary["scenes_done"] == 30
    assert summary["mean_seconds_per_site_year"] == 1500.0
    assert summary["eta_hours"] == 4 * 1500.0 / 2 / 3600.0


def test_summary_without_timings_has_no_eta():
    summary = status.summarise(sample_state(), {"total_rows": 0, "files": []}, 2, False)
    assert summary["eta_hours"] is None
    assert not summary["process_alive"]


def test_pid_alive_for_current_and_missing_process(tmp_path):
    alive = tmp_path / "alive.pid"
    alive.write_text(str(os.getpid()), encoding="utf-8")
    assert status.pid_alive(alive)
    assert not status.pid_alive(tmp_path / "missing.pid")
    broken = tmp_path / "broken.pid"
    broken.write_text("not a number", encoding="utf-8")
    assert not status.pid_alive(broken)


def test_remove_partial_files_only_removes_part_files(tmp_path):
    (tmp_path / "tidy").mkdir()
    (tmp_path / "tidy" / "x.parquet.part").write_text("x", encoding="utf-8")
    (tmp_path / "tidy" / "x.parquet").write_text("x", encoding="utf-8")
    removed = status.remove_partial_files(tmp_path)
    assert [path.name for path in removed] == ["x.parquet.part"]
    assert (tmp_path / "tidy" / "x.parquet").exists()
