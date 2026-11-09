from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rivwq.gee import connectivity_rules as rules
from rivwq.gee import export, roi, session

AREA_FILE = "areas.csv"
TEXT = {"site_no": str}


def variant_parameters(name: str, settings: dict) -> tuple[int, float]:
    if name == rules.CURRENT:
        return 0, settings["current_path_m"]
    kind, tag = name.split("_k")
    path = settings["cap_path_m"] if kind == "B" else settings["current_path_m"]
    return int(tag), path


def variant_areas(config: dict, settings: dict, point, k: int, path_m: float):
    import ee

    occurrence = ee.Image(config["collections"]["surface_water"]).select("occurrence").unmask(0)
    projection = occurrence.projection()
    water = occurrence.gte(config["masks"]["water_occurrence_min"])
    closed = water
    if k > 0:
        kernel = ee.Kernel.square(radius=k, units="pixels")
        closed = water.focalMax(kernel=kernel).focalMin(kernel=kernel)
    largest = max(config["roi"]["variants"].values())
    circle = point.buffer(largest)
    near = ee.Image(0).paint(
        ee.FeatureCollection([ee.Feature(point.buffer(config["masks"]["gauge_seed_distance_m"]))]),
        1,
    )
    seed = water.selfMask().updateMask(near)
    reached = (
        closed.selfMask().clip(circle).cumulativeCost(source=seed.unmask(0), maxDistance=path_m)
    )
    final = water.And(reached.mask()).selfMask()
    area = ee.Image.pixelArea().updateMask(final)
    result = {}
    for name, radius in config["roi"]["variants"].items():
        result[name] = area.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=point.buffer(radius),
            crs=projection,
            scale=settings["scale_m"],
            maxPixels=config["max_pixels"],
            tileScale=config["tile_scale"],
        ).get("area")
    return ee.Dictionary(result)


def read_cache(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=TEXT)
    return pd.DataFrame(columns=["variant", "site_no", "roi", "area_km2"])


def run_survey(config: dict, settings: dict, cache_dir: Path) -> pd.DataFrame:
    session.initialize()
    gauges = roi.load_gauges(roi.REPO_ROOT / config["roi"]["gauges_file"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / AREA_FILE
    frame = read_cache(path)
    done = set(zip(frame["variant"], frame["site_no"], strict=True))
    for name in rules.variant_list(settings):
        k, path_m = variant_parameters(name, settings)
        for site in sorted(gauges):
            if (name, site) in done:
                continue
            point = export.point_for(gauges, site)
            areas = variant_areas(config, settings, point, k, path_m)
            values = export.with_retries(lambda areas=areas: areas.getInfo(), config["retry"])
            rows = [
                {
                    "variant": name,
                    "site_no": site,
                    "roi": radius_name,
                    "area_km2": (value or 0.0) / 1e6,
                }
                for radius_name, value in values.items()
            ]
            frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
            frame.to_csv(path, index=False)
            print(f"done {name} {site}", flush=True)
    return frame


def plot_ratios(table: pd.DataFrame, settings: dict, names: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = [rules.CURRENT] + [rules.variant_name("A", k) for k in settings["gap_join_pixels"]]
    order = table[table["variant"] == rules.CURRENT].sort_values("ratio_20_5")["site_no"].tolist()
    figure, axis = plt.subplots(figsize=(8, 9))
    for variant in variants:
        subset = table[table["variant"] == variant].set_index("site_no").reindex(order)
        axis.scatter(subset["ratio_20_5"], range(len(order)), label=variant, s=22)
    axis.axvline(settings["thresholds"]["broken_ratio"], color="grey", linestyle="--")
    axis.axvline(settings["thresholds"]["leakage_ratio"], color="grey", linestyle=":")
    axis.set_xscale("log")
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels([f"{site} {names.get(site, '')}" for site in order], fontsize=7)
    axis.set_xlabel("area within 20 km / area within 5 km (log scale)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def station_names() -> dict:
    import yaml

    frozen = yaml.safe_load((roi.REPO_ROOT / "config" / "sites.yaml").read_text(encoding="utf-8"))
    return {
        item["site_no"]: item["station"][:22]
        for items in frozen["layers"].values()
        for item in items
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = export.load_config()
    settings = rules.load_config()
    frame = run_survey(config, settings, roi.REPO_ROOT / settings["cache_dir"])
    table = rules.wide_table(frame, settings)
    choice = rules.choose_k(table, settings)
    cap = rules.decide_cap(table, settings, choice["chosen_k"])
    args.report_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.report_dir / "05c_roi_connectivity.csv", index=False)
    plot_ratios(table, settings, station_names(), args.report_dir / "05c_roi_connectivity.png")
    print(choice)
    print(cap)
