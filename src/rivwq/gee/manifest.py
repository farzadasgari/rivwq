from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def utc_now() -> str:
    return datetime.now(UTC).strftime(TIMESTAMP_FORMAT)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=False))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_entry(run_dir: Path, path: Path, fetched_at: str) -> dict:
    return {
        "path": str(path.relative_to(run_dir)),
        "sha256": file_sha256(path),
        "rows": parquet_rows(path),
        "bytes": path.stat().st_size,
        "fetched_at": fetched_at,
    }


def entry_is_valid(run_dir: Path, entry: dict) -> bool:
    path = run_dir / entry["path"]
    if not path.exists():
        return False
    try:
        return file_sha256(path) == entry["sha256"] and parquet_rows(path) == entry["rows"]
    except Exception:
        return False


def completed_names(run_dir: Path, manifest: dict) -> set[str]:
    return {
        Path(entry["path"]).stem for entry in manifest["files"] if entry_is_valid(run_dir, entry)
    }


def upsert_entry(manifest: dict, entry: dict) -> dict:
    kept = [item for item in manifest["files"] if item["path"] != entry["path"]]
    kept.append(entry)
    result = dict(manifest)
    result["files"] = sorted(kept, key=lambda item: item["path"])
    result["total_rows"] = sum(item["rows"] for item in result["files"])
    result["manifest_written_utc"] = utc_now()
    return result


def build_state(
    run_id: str, started_utc: str, jobs: list[str], done: set[str], failed: dict
) -> dict:
    pending = [name for name in jobs if name not in done and name not in failed]
    return {
        "run_id": run_id,
        "started_utc": started_utc,
        "updated_utc": utc_now(),
        "done": sorted(name for name in jobs if name in done),
        "pending": pending,
        "failed": failed,
    }


def carry_over(previous: dict | None, jobs: list[str], years: list[int]) -> dict:
    names = set(jobs)
    carried_failed: dict = {}
    phases: list = []
    if previous:
        names |= set(previous["done"]) | set(previous["pending"]) | set(previous["failed"])
        carried_failed = dict(previous["failed"])
        phases = list(previous.get("phases", []))
    phases.append({"years": years, "started_utc": utc_now()})
    return {"names": sorted(names), "failed": carried_failed, "phases": phases}


def snapshot(run_id: str, started_utc: str, carried: dict, done: set[str], failed: dict) -> dict:
    merged = {**carried["failed"], **failed}
    open_failures = {name: message for name, message in merged.items() if name not in done}
    state = build_state(run_id, started_utc, carried["names"], done, open_failures)
    state["phases"] = carried["phases"]
    return state
