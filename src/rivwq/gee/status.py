from __future__ import annotations

import os
from pathlib import Path


def pid_alive(pid_path: Path) -> bool:
    if not pid_path.exists():
        return False
    text = pid_path.read_text(encoding="utf-8").strip()
    if not text.isdigit():
        return False
    try:
        os.kill(int(text), 0)
    except OSError:
        return False
    return True


def summarise(state: dict, manifest: dict, workers: int, alive: bool) -> dict:
    timed = [item for item in manifest["files"] if item.get("seconds")]
    scenes = sum(item.get("scenes") or 0 for item in manifest["files"])
    mean_seconds = sum(item["seconds"] for item in timed) / len(timed) if timed else None
    pending = len(state["pending"])
    eta_hours = None
    if mean_seconds is not None:
        eta_hours = pending * mean_seconds / max(workers, 1) / 3600.0
    return {
        "run_id": state["run_id"],
        "process_alive": alive,
        "done": len(state["done"]),
        "pending": pending,
        "failed": len(state["failed"]),
        "scenes_done": scenes,
        "rows": manifest["total_rows"],
        "mean_seconds_per_site_year": mean_seconds,
        "eta_hours": eta_hours,
        "updated_utc": state["updated_utc"],
    }


def remove_partial_files(run_dir: Path) -> list[Path]:
    removed = sorted(run_dir.rglob("*.part"))
    for path in removed:
        path.unlink()
    return removed
