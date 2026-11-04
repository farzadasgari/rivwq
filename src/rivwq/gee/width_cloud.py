from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from rivwq.gee import screening, session

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "width_cloud.yaml"
SCOUT_CONFIG_PATH = REPO_ROOT / "config" / "scout.yaml"
TEXT = {"site_no": str}
RAW_WIDTH_FILE = "width_raw.csv"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_site_table(input_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(input_dir / name, dtype=TEXT).set_index("site_no")


def read_inputs(config: dict) -> pd.DataFrame:
    input_dir = Path(config["input_dir"]).expanduser()
    shortlist = read_site_table(input_dir, config["shortlist_file"])
    candidates = read_site_table(input_dir, config["candidates_file"])
    with open(SCOUT_CONFIG_PATH, encoding="utf-8") as handle:
        scout_config = yaml.safe_load(handle)
    return screening.select_sites(
        shortlist,
        candidates,
        scout_config["reference_sites"],
        config["extra_candidates"],
    )


def verify_datasets(config: dict) -> dict[str, str]:
    import ee

    checked = {}
    for name, asset_id in config["datasets"].items():
        info = ee.data.getAsset(asset_id)
        checked[name] = f"{asset_id} ({info['type']})"
    return checked


def gauge_points(sites: pd.DataFrame):
    import ee

    features = [
        ee.Feature(ee.Geometry.Point([row["dec_long_va"], row["dec_lat_va"]]), {"site_no": site})
        for site, row in sites.iterrows()
    ]
    return ee.FeatureCollection(features)


def width_feature_factory(config: dict):
    import ee

    settings = config["width"]
    occurrence = (
        ee.Image(config["datasets"]["surface_water"])
        .select(config["bands"]["occurrence"])
        .unmask(0)
    )
    area = ee.Image.pixelArea()
    layers = [area.rename("total_area_m2")]
    for threshold in settings["occurrence_thresholds"]:
        layers.append(
            area.updateMask(occurrence.gte(threshold))
            .unmask(0)
            .rename(f"water_area_m2_occ{threshold}")
        )
    stack = ee.Image.cat(layers)
    projection = occurrence.projection()

    def compute(feature):
        point = feature.geometry()
        sums = stack.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=point.buffer(settings["circle_radius_m"]),
            crs=projection,
            scale=settings["scale_m"],
            maxPixels=int(1e9),
        )
        near = occurrence.reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=point.buffer(settings["gauge_near_water_distance_m"]),
            crs=projection,
            scale=settings["scale_m"],
            maxPixels=int(1e9),
        ).get(config["bands"]["occurrence"])
        return feature.set(sums).set("max_occurrence_near_gauge", near)

    return compute


def fetch_raw_width(
    config: dict, sites: pd.DataFrame, cache: Path, batch_size: int | None = None
) -> pd.DataFrame:
    path = cache / RAW_WIDTH_FILE
    cached = pd.read_csv(path, dtype=TEXT).set_index("site_no") if path.exists() else None
    missing = sites if cached is None else sites.loc[~sites.index.isin(cached.index)]
    if len(missing):
        step = batch_size or len(missing)
        frames = [] if cached is None else [cached]
        compute = width_feature_factory(config)
        for begin in range(0, len(missing), step):
            batch = missing.iloc[begin : begin + step]
            collection = gauge_points(batch).map(compute)
            features = with_retries(
                lambda collection=collection: collection.getInfo()["features"],
                config["cloud"]["retry_attempts"],
                config["cloud"]["retry_pause_seconds"],
            )
            frames.append(
                pd.DataFrame([item["properties"] for item in features]).set_index("site_no")
            )
        cached = pd.concat(frames)
        cache.mkdir(parents=True, exist_ok=True)
        cached.to_csv(path)
    return cached.loc[sites.index]


def per_date_collection(config: dict, longitude: float, latitude: float, year: int):
    import ee

    cloud = config["cloud"]
    bands = config["bands"]
    roi = ee.Geometry.Point([longitude, latitude]).buffer(cloud["roi_buffer_m"])
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    clouds = (
        ee.ImageCollection(config["datasets"]["cloud_score"])
        .filterBounds(roi)
        .filterDate(start, end)
    )
    scenes = (
        ee.ImageCollection(config["datasets"]["surface_reflectance"])
        .filterBounds(roi)
        .filterDate(start, end)
        .linkCollection(clouds, [bands["cloud_score"]])
    )
    days = (
        ee.List(scenes.aggregate_array("system:time_start"))
        .map(lambda stamp: ee.Date(stamp).format("YYYY-MM-dd"))
        .distinct()
    )
    classes = cloud["scl_valid_classes"]

    def summarise_day(day):
        day = ee.String(day)
        begin = ee.Date(day)
        images = scenes.filterDate(begin, begin.advance(1, "day"))
        valid_cs = images.map(
            lambda image: image.select(bands["cloud_score"]).gte(cloud["cloud_score_min"])
        ).max()
        valid_scl = images.map(
            lambda image: image.select(bands["scene_classification"]).remap(
                classes, [1] * len(classes), 0
            )
        ).max()
        covered = images.map(lambda image: image.select(bands["coverage"]).mask()).max()
        stack = ee.Image.cat(
            [valid_cs.rename("valid_cs"), valid_scl.rename("valid_scl"), covered.rename("covered")]
        ).unmask(0)
        means = stack.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=cloud["scale_m"],
            maxPixels=int(1e9),
            tileScale=cloud["tile_scale"],
        )
        return ee.Feature(None, means).set({"date": day, "scenes": images.size()})

    return ee.FeatureCollection(days.map(summarise_day))


def fetch_site_year(config: dict, site: str, row: pd.Series, year: int, cache: Path) -> Path:
    path = cache / "cloud" / f"{site}_{year}.csv"
    if path.exists():
        return path
    cloud = config["cloud"]
    collection = per_date_collection(config, row["dec_long_va"], row["dec_lat_va"], year)
    features = with_retries(
        lambda: collection.getInfo()["features"],
        cloud["retry_attempts"],
        cloud["retry_pause_seconds"],
    )
    columns = ["date", "scenes", "valid_cs", "valid_scl", "covered"]
    frame = pd.DataFrame([item["properties"] for item in features], columns=columns)
    frame.insert(0, "site_no", site)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def with_retries(call, attempts: int, pause: float):
    for attempt in range(attempts):
        try:
            return call()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(pause * 2**attempt)


def pending_jobs(sites: pd.DataFrame, years: list[int], cache: Path) -> list[tuple[str, int]]:
    return [
        (site, year)
        for site in sites.index
        for year in years
        if not (cache / "cloud" / f"{site}_{year}.csv").exists()
    ]


def fetch_cloud(config: dict, sites: pd.DataFrame, years: list[int], cache: Path) -> pd.DataFrame:
    jobs = pending_jobs(sites, years, cache)
    print(f"cloud jobs pending: {len(jobs)} of {len(sites) * len(years)}")
    with ThreadPoolExecutor(max_workers=config["cloud"]["workers"]) as pool:
        futures = {
            job: pool.submit(fetch_site_year, config, job[0], sites.loc[job[0]], job[1], cache)
            for job in jobs
        }
        for job, future in futures.items():
            future.result()
            print(f"done {job[0]} {job[1]}")
    frames = [
        pd.read_csv(cache / "cloud" / f"{site}_{year}.csv", dtype=TEXT)
        for site in sites.index
        for year in years
    ]
    return pd.concat(frames, ignore_index=True)


def write_manifest(config: dict, checked: dict, site_count: int, path: Path) -> None:
    import ee

    manifest = {
        "run_date": date.today().isoformat(),
        "earth_engine_api_version": ee.__version__,
        "site_count": site_count,
        "datasets_verified": checked,
        "width": config["width"],
        "cloud": config["cloud"],
        "bands": config["bands"],
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    session.initialize()
    cache = REPO_ROOT / config["cache_dir"]
    output_dir = Path(config["output_dir"]).expanduser()
    prefix = config["output_prefix"]
    sites = read_inputs(config)
    checked = verify_datasets(config)
    raw = fetch_raw_width(config, sites, cache)
    first_year = int(config["cloud"]["start"][:4])
    last_year = int(config["cloud"]["end_exclusive"][:4]) - 1
    years = list(range(first_year, last_year + 1))
    per_date = fetch_cloud(config, sites, years, cache)
    width = screening.width_table(raw, config["width"])
    cloud = screening.cloud_summary(
        per_date, list(sites.index), years, config["cloud"]["min_valid_roi_fraction"]
    )
    table = screening.assemble(sites, width, cloud)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / f"{prefix}.csv")
    write_manifest(config, checked, len(table), output_dir / f"{prefix}_manifest.json")
    print(output_dir / f"{prefix}.csv")
