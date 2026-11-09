from __future__ import annotations

import numpy as np
import pandas as pd

REDUCER_SUFFIX = {"mean": "mean", "median": "median", "stdDev": "std"}
SCENE_KEY = "system_index"


def cloud_tags(config: dict) -> list[str]:
    thresholds = config["masks"]["cloud_score_thresholds"]
    return [f"cs{round(value * 100):02d}" for value in thresholds] + ["scl"]


def erosion_tag(pixels: int) -> str:
    return f"e{pixels}"


def band_list(config: dict) -> list[str]:
    return list(config["bands"]["reflectance"]) + list(config["bands"]["atmosphere"])


def column_name(roi: str, erosion: int, tag: str, band: str, reducer: str) -> str:
    return f"{roi}__{erosion_tag(erosion)}__{tag}__{band}_{reducer}"


def count_name(roi: str, erosion: int, tag: str) -> str:
    return f"{roi}__{erosion_tag(erosion)}__{tag}__valid_count"


def total_name(roi: str, erosion: int) -> str:
    return f"{roi}__{erosion_tag(erosion)}__total_count"


def cloud_valid_masks(
    cloud_score: np.ndarray, classification: np.ndarray, config: dict
) -> dict[str, np.ndarray]:
    masks = {}
    for threshold, tag in zip(
        config["masks"]["cloud_score_thresholds"], cloud_tags(config)[:-1], strict=True
    ):
        masks[tag] = cloud_score >= threshold
    masks["scl"] = np.isin(classification, config["masks"]["scl_valid_classes"])
    return masks


def erode_plus(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    centre = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return centre & up & down & left & right


def masked_statistics(values: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    selected = values[valid]
    if selected.size == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan, "count": 0}
    return {
        "mean": float(selected.mean()),
        "median": float(np.median(selected)),
        "std": float(selected.std(ddof=0)),
        "count": int(selected.size),
    }


def twin_scene_statistics(
    bands: dict[str, np.ndarray],
    cloud_score: np.ndarray,
    classification: np.ndarray,
    water: np.ndarray,
    footprint: np.ndarray,
    config: dict,
) -> dict[tuple[int, str], dict]:
    clouds = cloud_valid_masks(cloud_score, classification, config)
    result = {}
    for erosion in config["masks"]["erosion_pixels"]:
        base = water.astype(bool)
        for _ in range(erosion):
            base = erode_plus(base)
        base = base & footprint
        for tag, cloud in clouds.items():
            valid = base & cloud
            entry = {"total_count": int(base.sum()), "valid_count": int(valid.sum())}
            for band, values in bands.items():
                entry[band] = masked_statistics(values, valid)
            result[(erosion, tag)] = entry
    return result


def valid_fraction(valid: pd.Series, total: pd.Series) -> pd.Series:
    return (valid / total.where(total > 0)).astype(float)


def spacecraft_code(name: str, mapping: dict[str, str]) -> str:
    if name not in mapping:
        raise KeyError(f"unknown spacecraft name: {name}")
    return mapping[name]


def scene_table(features: list[dict], config: dict) -> pd.DataFrame:
    rename = config["scene_properties"]
    rows = []
    for feature in features:
        properties = feature["properties"]
        row = {SCENE_KEY: properties["system_index"]}
        for source, target in rename.items():
            row[target] = properties.get(source)
        row["has_cloud_score"] = bool(properties.get("has_cloud_score"))
        for key, value in properties.items():
            if "__" in key:
                row[key] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["spacecraft"] = frame["spacecraft"].map(
        lambda name: spacecraft_code(name, config["spacecraft"])
    )
    frame["time_utc"] = pd.to_datetime(frame["time_start_ms"], unit="ms", utc=True)
    return frame


META_COLUMNS = [
    SCENE_KEY,
    "time_utc",
    "time_start_ms",
    "mgrs_tile",
    "relative_orbit",
    "spacecraft",
    "processing_baseline",
    "solar_zenith",
    "solar_azimuth",
    "view_zenith_b8a",
    "view_azimuth_b8a",
    "cloudy_pixel_percentage",
    "has_cloud_score",
]


def long_columns(config: dict) -> list[str]:
    value_columns = [
        f"{band}_{reducer}" for band in band_list(config) for reducer in REDUCER_SUFFIX.values()
    ]
    keys = ["site_no", "roi", "erosion_px", "cloud_variant"]
    counts = ["total_count", "valid_count", "valid_fraction"]
    return META_COLUMNS + keys + counts + value_columns


def to_long(
    wide: pd.DataFrame, site: str, config: dict, rois: list[str] | None = None
) -> pd.DataFrame:
    meta_columns = META_COLUMNS
    bands = band_list(config)
    reducers = list(REDUCER_SUFFIX.values())
    pieces = []
    for roi in rois or list(config["roi"]["variants"]):
        for erosion in config["masks"]["erosion_pixels"]:
            for tag in cloud_tags(config):
                piece = wide[meta_columns].copy()
                piece["site_no"] = site
                piece["roi"] = roi
                piece["erosion_px"] = erosion
                piece["cloud_variant"] = tag
                piece["total_count"] = wide[total_name(roi, erosion)]
                piece["valid_count"] = wide[count_name(roi, erosion, tag)]
                piece["valid_fraction"] = valid_fraction(piece["valid_count"], piece["total_count"])
                for band in bands:
                    for reducer in reducers:
                        piece[f"{band}_{reducer}"] = wide[
                            column_name(roi, erosion, tag, band, reducer)
                        ]
                pieces.append(piece)
    long = pd.concat(pieces, ignore_index=True)
    return long.sort_values(["time_utc", SCENE_KEY, "roi", "erosion_px", "cloud_variant"])
