from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from rivwq.usgs import download
from rivwq.usgs.qc_rules import REPO_ROOT, load_config
from rivwq.usgs.tidy import tidy_response

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
FETCHED_SOURCE = "response cache file mtime"


def site_year_name(entry_path: str) -> str:
    return Path(entry_path).stem


def fetched_at(response_path: Path) -> str:
    modified = datetime.fromtimestamp(response_path.stat().st_mtime, tz=UTC)
    return modified.strftime(TIMESTAMP_FORMAT)


def add_fetched_at(manifest: dict, run_dir: Path, refetched: list[str]) -> dict:
    result = dict(manifest)
    result["files"] = []
    for entry in manifest["files"]:
        name = site_year_name(entry["path"])
        site = name.split("_")[0]
        response_path = run_dir / "responses" / site / f"{name}.parquet"
        updated = dict(entry)
        updated["fetched_at"] = fetched_at(response_path)
        updated["refetched"] = name in refetched
        result["files"].append(updated)
    result["fetched_at_source"] = FETCHED_SOURCE
    return result


def candidate_site_years(manifest: dict, refetched: list[str], last_year: int) -> list[str]:
    names = [site_year_name(entry["path"]) for entry in manifest["files"]]
    return [name for name in names if name not in refetched and int(name.split("_")[1]) < last_year]


def pick_site_years(candidates: list[str], seed: int, count: int) -> list[str]:
    return sorted(random.Random(seed).sample(sorted(candidates), count))


def stored_codes(run_dir: Path, name: str) -> tuple[str, ...]:
    site = name.split("_")[0]
    response = pd.read_parquet(run_dir / "responses" / site / f"{name}.parquet")
    return tuple(sorted(response["parameter_code"].astype(str).unique()))


def compare_refetch(run_dir: Path, name: str, scratch: Path, config: dict, pacer) -> dict:
    site, year = name.split("_")
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    run_end = pd.Timestamp(state["started_utc"])
    codes = stored_codes(run_dir, name)
    response = download.fetch_site_year(site, int(year), codes, run_end, config, pacer)
    download.write_parquet(response, scratch / "responses" / site / f"{name}.parquet")
    fresh = tidy_response(response, config["source_column"])
    download.write_parquet(fresh, scratch / "tidy" / site / f"{name}.parquet")
    stored = pd.read_parquet(run_dir / "tidy" / site / f"{name}.parquet")
    equal = fresh.equals(stored)
    return {
        "site_year": name,
        "stored_rows": len(stored),
        "fresh_rows": len(fresh),
        "equal": bool(equal),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--skip-refetch", action="store_true")
    args = parser.parse_args(argv)
    config = load_config()
    settings = config["manifest_check"]
    run_dir = REPO_ROOT / config["raw_root"] / args.run_id
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    refetched = settings["refetched_site_years"]
    scratch = Path(settings["scratch_dir"]).expanduser()
    scratch.mkdir(parents=True, exist_ok=True)
    if "fetched_at_source" not in manifest:
        backup = scratch / "manifest_before_fetched_at.json"
        backup.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        hardened = add_fetched_at(manifest, run_dir, refetched)
        manifest_path.write_text(json.dumps(hardened, indent=2), encoding="utf-8")
        print(f"fetched_at added to {len(hardened['files'])} files")
    if args.skip_refetch:
        return
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    last_year = pd.Timestamp(state["started_utc"]).year
    candidates = candidate_site_years(manifest, refetched, last_year)
    chosen = pick_site_years(candidates, settings["seed"], settings["site_years"])
    download_config = download.load_config()
    pacer = download.RequestPacer(download_config["min_request_interval_seconds"])
    results = [compare_refetch(run_dir, n, scratch, download_config, pacer) for n in chosen]
    (scratch / "refetch_comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
