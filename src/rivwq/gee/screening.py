from __future__ import annotations

import numpy as np
import pandas as pd

BASE_COLUMNS = [
    "station_nm",
    "state",
    "dec_lat_va",
    "dec_long_va",
    "drainage_area_sqmi",
    "turbidity_class",
    "regulation_class",
    "regime_class",
    "sampled_day_fraction",
    "fchl_first",
    "fchl_last",
    "fchl_years",
    "fchl_before_2025",
    "turbidity_only",
    "caveat",
    "caveat_reason",
]


def select_sites(
    shortlist: pd.DataFrame,
    candidates: pd.DataFrame,
    reference_sites: list[str],
    extra_sites: list[str],
) -> pd.DataFrame:
    roles = {site: "shortlist" for site in shortlist.index}
    roles.update({site: "extra" for site in extra_sites})
    roles.update({site: "reference" for site in reference_sites})
    missing = [site for site in roles if site not in candidates.index]
    if missing:
        raise KeyError(f"sites missing from candidates: {missing}")
    sites = candidates.loc[list(roles), BASE_COLUMNS].copy()
    sites.insert(0, "role", pd.Series(roles))
    return sites


def pixels_across(width_m: float, pixel_size_m: float) -> float:
    return width_m / pixel_size_m


def is_lake_like(water_share: float, minimum_share: float) -> bool:
    return bool(water_share >= minimum_share)


def gauge_near_water(max_occurrence_nearby: float, threshold: float) -> bool:
    if np.isnan(max_occurrence_nearby):
        return False
    return bool(max_occurrence_nearby >= threshold)


def width_metrics(raw: dict, settings: dict) -> dict:
    diameter = 2.0 * settings["circle_radius_m"]
    primary = settings["primary_occurrence_threshold"]
    total = raw["total_area_m2"]
    result = {}
    for threshold in settings["occurrence_thresholds"]:
        area = raw[f"water_area_m2_occ{threshold}"]
        result[f"width_m_occ{threshold}"] = area / diameter
        result[f"water_share_occ{threshold}"] = area / total
    width = result[f"width_m_occ{primary}"]
    share = result[f"water_share_occ{primary}"]
    near = raw["max_occurrence_near_gauge"]
    result["width_m"] = width
    result["pixels_across"] = pixels_across(width, settings["pixel_size_m"])
    result["water_share"] = share
    result["max_occurrence_near_gauge"] = near
    result["gauge_near_water"] = gauge_near_water(near, primary)
    result["lake_like"] = is_lake_like(share, settings["lake_like_min_water_share"])
    result["under_min_pixels_across"] = result["pixels_across"] < settings["min_pixels_across"]
    return result


def width_table(raw: pd.DataFrame, settings: dict) -> pd.DataFrame:
    rows = {site: width_metrics(row.to_dict(), settings) for site, row in raw.iterrows()}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("site_no")


def valid_date_share(per_date: pd.DataFrame, column: str, minimum_fraction: float) -> float:
    if per_date.empty:
        return np.nan
    return float((per_date[column] >= minimum_fraction).mean())


def cloud_summary(
    per_date: pd.DataFrame, sites: list[str], years: list[int], minimum_fraction: float
) -> pd.DataFrame:
    rows = {}
    frame = per_date.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    for site in sites:
        site_frame = frame[frame["site_no"] == site]
        row = {}
        counts = []
        for year in years:
            year_frame = site_frame[site_frame["year"] == year]
            row[f"dates_{year}"] = len(year_frame)
            counts.append(len(year_frame))
            row[f"cloud_score_share_{year}"] = valid_date_share(
                year_frame, "valid_cs", minimum_fraction
            )
            row[f"scl_share_{year}"] = valid_date_share(year_frame, "valid_scl", minimum_fraction)
        row["dates_per_year_mean"] = float(np.mean(counts))
        row["dates_total"] = int(np.sum(counts))
        row["cloud_score_share"] = valid_date_share(site_frame, "valid_cs", minimum_fraction)
        row["scl_share"] = valid_date_share(site_frame, "valid_scl", minimum_fraction)
        row["dates_with_roi_data_share"] = (
            float((site_frame["covered"] > 0).mean()) if len(site_frame) else np.nan
        )
        rows[site] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("site_no")


def assemble(sites: pd.DataFrame, width: pd.DataFrame, cloud: pd.DataFrame) -> pd.DataFrame:
    table = sites.join(width).join(cloud)
    return table.sort_values("drainage_area_sqmi").rename_axis("site_no")
