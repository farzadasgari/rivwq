from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_gauges(path: Path) -> dict[str, tuple[float, float]]:
    collection = json.loads(path.read_text(encoding="utf-8"))
    gauges = {}
    for feature in collection["features"]:
        longitude, latitude = feature["geometry"]["coordinates"]
        gauges[feature["properties"]["site_no"]] = (float(longitude), float(latitude))
    return gauges


def roi_hash(
    gauges: dict[str, tuple[float, float]], config: dict, skips: dict[str, list[str]] | None = None
) -> str:
    payload = {
        "gauges": {site: list(point) for site, point in sorted(gauges.items())},
        "variants": config["roi"]["variants"],
        "masks": config["masks"],
        "scale_m": config["scale_m"],
    }
    if skips:
        payload["skipped_rois"] = {site: sorted(names) for site, names in sorted(skips.items())}
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utm_epsg(longitude: float) -> str:
    zone = int((longitude + 180.0) // 6.0) + 1
    return f"EPSG:326{zone:02d}"


def largest_radius(config: dict) -> float:
    return float(max(config["roi"]["variants"].values()))
