from __future__ import annotations

import argparse
import gzip
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from rivwq.gee import extract, manifest, roi, session, stats, status

REPO_ROOT = roi.REPO_ROOT
CONFIG_PATH = REPO_ROOT / "config" / "gee.yaml"
QUOTA_MARKERS = ("quota", "too many requests", "429", "rate limit")
ROI_AREAS_NAME = "roi_areas.parquet"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def is_quota_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in QUOTA_MARKERS)


def with_retries(call, retry: dict, sleep=time.sleep):
    ordinary = 0
    quota = 0
    while True:
        try:
            return call()
        except Exception as error:
            if is_quota_error(error):
                quota += 1
                if quota >= retry["quota_attempts"]:
                    raise
                sleep(retry["quota_wait_seconds"])
                continue
            ordinary += 1
            if ordinary >= retry["attempts"]:
                raise
            sleep(retry["backoff_seconds"] * 2 ** (ordinary - 1))


def verify_assets(config: dict) -> dict:
    import ee

    verified = {}
    for name, asset_id in config["collections"].items():
        info = ee.data.getAsset(asset_id)
        verified[name] = {
            "asset_id": asset_id,
            "type": info["type"],
            "start_time": info.get("startTime"),
            "end_time": info.get("endTime"),
            "update_time": info.get("updateTime"),
        }
    water = verified["surface_water"]
    if water["type"] != "IMAGE":
        raise RuntimeError("surface water asset is not a single static image")
    return verified


def year_bounds(year: int, date_start: str, run_start: datetime) -> tuple[str, str]:
    begin = max(
        datetime(year, 1, 1, tzinfo=UTC), datetime.fromisoformat(date_start).replace(tzinfo=UTC)
    )
    end = min(datetime(year + 1, 1, 1, tzinfo=UTC), run_start)
    return begin.strftime(manifest.TIMESTAMP_FORMAT), end.strftime(manifest.TIMESTAMP_FORMAT)


def job_name(site: str, year: int) -> str:
    return f"{site}_{year}"


def plan_jobs(sites: list[str], years: list[int]) -> list[tuple[str, int]]:
    return [(site, year) for site in sites for year in years]


def point_for(gauges: dict, site: str):
    import ee

    longitude, latitude = gauges[site]
    return ee.Geometry.Point([longitude, latitude])


def batches(items: list, size: int) -> list[list]:
    return [items[begin : begin + size] for begin in range(0, len(items), size)]


def load_geometries(config: dict, gauges: dict, site: str, run_dir: Path) -> dict:
    path = run_dir / "roi_geometries" / f"{site}.json"
    if path.exists():
        return manifest.read_json(path)
    polygons = extract.water_polygons(config, point_for(gauges, site))
    geometries = {
        name: with_retries(lambda polygon=polygon: polygon.getInfo(), config["retry"])
        for name, polygon in polygons.items()
    }
    manifest.atomic_write_json(path, geometries)
    return geometries


def fetch_features(
    config: dict, gauges: dict, site: str, year: int, run_start: datetime, geometries: dict
) -> list:
    start, end = year_bounds(year, config["date_start"], run_start)
    point = point_for(gauges, site)
    ids = with_retries(lambda: extract.scene_ids(config, point, start, end), config["retry"])
    features = []
    for chunk in batches(ids, config["chunking"]["scenes_per_request"]):
        collection = extract.scene_batch_collection(config, point, start, end, chunk, geometries)

        def call(collection=collection) -> list:
            return collection.getInfo()["features"]

        features.extend(with_retries(call, config["retry"]))
    if len(features) != len(ids):
        raise RuntimeError(f"{site} {year}: expected {len(ids)} scenes, received {len(features)}")
    return features


def write_raw(features: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(features, handle)
    os.replace(temporary, path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def empty_long(config: dict) -> pd.DataFrame:
    return pd.DataFrame(columns=stats.long_columns(config))


def tidy_site_year(features: list, site: str, config: dict) -> pd.DataFrame:
    if not features:
        return empty_long(config)
    wide = stats.scene_table(features, config)
    return stats.to_long(wide, site, config)


def process_site_year(
    config: dict, gauges: dict, site: str, year: int, run_dir: Path, run_start: datetime
) -> dict:
    began = time.monotonic()
    geometries = load_geometries(config, gauges, site, run_dir)
    features = fetch_features(config, gauges, site, year, run_start, geometries)
    name = job_name(site, year)
    write_raw(features, run_dir / "responses" / site / f"{name}.json.gz")
    tidy = tidy_site_year(features, site, config)
    path = run_dir / "tidy" / site / f"{name}.parquet"
    write_parquet(tidy, path)
    entry = manifest.file_entry(run_dir, path, manifest.utc_now())
    entry["scenes"] = len(features)
    entry["seconds"] = round(time.monotonic() - began, 1)
    return entry


def compute_roi_areas(config: dict, gauges: dict, sites: list[str]) -> pd.DataFrame:
    rows = []
    for site in sites:
        longitude, _ = gauges[site]
        areas = extract.roi_area_table(config, point_for(gauges, site), roi.utm_epsg(longitude))
        values = with_retries(lambda areas=areas: areas.getInfo(), config["retry"])
        for key, value in values.items():
            variant, erosion = key.split("__")
            rows.append(
                {
                    "site_no": site,
                    "roi": variant,
                    "erosion_px": int(erosion.removeprefix("e")),
                    "water_area_m2": value,
                }
            )
    return pd.DataFrame(rows)


def initial_manifest(
    config: dict, run_id: str, run_start: datetime, gauges: dict, sites: list
) -> dict:
    import ee

    return {
        "run_id": run_id,
        "run_started_utc": run_start.strftime(manifest.TIMESTAMP_FORMAT),
        "earth_engine_api_version": ee.__version__,
        "assets": verify_assets(config),
        "filters": {
            "bounds": "circle of the largest ROI radius around the gauge",
            "date": [config["date_start"], run_start.strftime(manifest.TIMESTAMP_FORMAT)],
            "cloud_score_join": (
                "saveFirst on system:index; scenes without a match kept, has_cloud_score false"
            ),
        },
        "roi_hash": roi.roi_hash(gauges, config),
        "roi_variants_m": config["roi"]["variants"],
        "sites": sites,
        "bands": stats.band_list(config),
        "band_units": "raw L2A digital numbers, reflectance = DN / 10000, not scaled",
        "reducers": config["reducers"],
        "scale_m": config["scale_m"],
        "crs": config["crs"],
        "tile_scale": config["tile_scale"],
        "masks": config["masks"],
        "water_mask_note": (
            "static Global Surface Water occurrence image (1984-2021), same for every scene"
        ),
        "total_rows": 0,
        "files": [],
    }


def run(
    config: dict, run_id: str, sites: list[str], years: list[int], raw_root: Path | None = None
) -> dict:
    session.initialize()
    gauges = roi.load_gauges(REPO_ROOT / config["roi"]["gauges_file"])
    run_dir = (raw_root or REPO_ROOT / config["raw_root"]) / run_id
    manifest_path = run_dir / "manifest.json"
    state_path = run_dir / "run_state.json"
    if manifest_path.exists():
        current = manifest.read_json(manifest_path)
        run_start = datetime.strptime(
            current["run_started_utc"], manifest.TIMESTAMP_FORMAT
        ).replace(tzinfo=UTC)
        if current["roi_hash"] != roi.roi_hash(gauges, config):
            raise RuntimeError("ROI hash differs from the run being resumed")
    else:
        run_start = datetime.now(UTC).replace(microsecond=0)
        current = initial_manifest(config, run_id, run_start, gauges, sites)
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest.atomic_write_json(manifest_path, current)
    for leftover in status.remove_partial_files(run_dir):
        print(f"removed partial file {leftover}")
    jobs = plan_jobs(sites, years)
    names = [job_name(*job) for job in jobs]
    done = manifest.completed_names(run_dir, current)
    failed: dict = {}
    lock = threading.Lock()
    started = current["run_started_utc"]
    manifest.atomic_write_json(
        state_path, manifest.build_state(run_id, started, names, done, failed)
    )
    areas_path = run_dir / ROI_AREAS_NAME
    areas_entry_ok = any(
        item["path"] == ROI_AREAS_NAME and manifest.entry_is_valid(run_dir, item)
        for item in current["files"]
    )
    if not areas_entry_ok:
        write_parquet(compute_roi_areas(config, gauges, sites), areas_path)
        current = manifest.upsert_entry(
            current, manifest.file_entry(run_dir, areas_path, manifest.utc_now())
        )
        manifest.atomic_write_json(manifest_path, current)

    def work(job: tuple[str, int]) -> None:
        site, year = job
        name = job_name(site, year)
        try:
            entry = process_site_year(config, gauges, site, year, run_dir, run_start)
        except Exception as error:
            with lock:
                failed[name] = f"{type(error).__name__}: {error}"
                manifest.atomic_write_json(
                    state_path, manifest.build_state(run_id, started, names, done, failed)
                )
            return
        with lock:
            nonlocal current
            current = manifest.upsert_entry(current, entry)
            done.add(name)
            manifest.atomic_write_json(manifest_path, current)
            manifest.atomic_write_json(
                state_path, manifest.build_state(run_id, started, names, done, failed)
            )
        print(f"done {name}: {entry['scenes']} scenes, {entry['seconds']} s", flush=True)

    todo = [job for job in jobs if job_name(*job) not in done]
    print(f"site-years: {len(jobs)}, already done: {len(jobs) - len(todo)}, to fetch: {len(todo)}")
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        list(pool.map(work, todo))
    if failed:
        print(f"FAILED site-years: {len(failed)}")
        for name, message in failed.items():
            print(f"  {name}: {message}")
    return current


def all_sites(config: dict) -> list[str]:
    return sorted(roi.load_gauges(REPO_ROOT / config["roi"]["gauges_file"]))


def all_years(run_start: datetime, config: dict) -> list[int]:
    return list(range(int(config["date_start"][:4]), run_start.year + 1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    today = datetime.now(UTC)
    if args.mode == "pilot":
        run_id = args.run_id or f"gee_pilot_{today:%Y%m%d}"
        run(config, run_id, config["pilot"]["sites"], config["pilot"]["years"])
        return
    if not args.run_id:
        raise SystemExit("full mode needs --run-id so a resume reuses the same run")
    run(config, args.run_id, all_sites(config), all_years(today, config))
