from __future__ import annotations

from pathlib import Path

import pandas as pd

from rivwq.gee import manifest


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def entry_for(run_dir: Path, name: str, rows: int) -> dict:
    path = run_dir / "tidy" / "1" / f"{name}.parquet"
    write_parquet(pd.DataFrame({"a": range(rows)}), path)
    return manifest.file_entry(run_dir, path, "2026-11-08T00:00:00Z")


def test_resume_skips_valid_and_refetches_changed_or_missing(tmp_path):
    first = entry_for(tmp_path, "1_2023", 3)
    second = entry_for(tmp_path, "1_2024", 4)
    third = entry_for(tmp_path, "1_2025", 5)
    state = {"files": []}
    for entry in (first, second, third):
        state = manifest.upsert_entry(state, entry)
    assert manifest.completed_names(tmp_path, state) == {"1_2023", "1_2024", "1_2025"}
    (tmp_path / second["path"]).unlink()
    write_parquet(pd.DataFrame({"a": [9, 9, 9, 9, 9]}), tmp_path / third["path"])
    assert manifest.completed_names(tmp_path, state) == {"1_2023"}


def test_upsert_replaces_entry_and_updates_total_rows(tmp_path):
    state = {"files": []}
    state = manifest.upsert_entry(state, entry_for(tmp_path, "1_2023", 3))
    state = manifest.upsert_entry(state, entry_for(tmp_path, "1_2023", 5))
    assert len(state["files"]) == 1
    assert state["total_rows"] == 5


def test_state_lists_done_pending_failed():
    state = manifest.build_state("r", "t", ["a", "b", "c"], {"a"}, {"b": "error"})
    assert state["done"] == ["a"]
    assert state["pending"] == ["c"]
    assert state["failed"] == {"b": "error"}


def test_atomic_json_write_leaves_no_partial_file(tmp_path):
    path = tmp_path / "run" / "state.json"
    manifest.atomic_write_json(path, {"a": 1})
    assert manifest.read_json(path) == {"a": 1}
    assert not list(path.parent.glob("*.part"))
