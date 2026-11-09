from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rivwq.gee import stats


def load_run(run_dir: Path) -> pd.DataFrame:
    paths = sorted((run_dir / "tidy").rglob("*.parquet"))
    frames = [pd.read_parquet(path) for path in paths]
    frames = [frame for frame in frames if len(frame)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def reference_rows(long: pd.DataFrame, settings: dict) -> pd.DataFrame:
    keep = (
        (long["roi"] == settings["reference_roi"])
        & (long["erosion_px"] == settings["reference_erosion"])
        & (long["cloud_variant"] == settings["reference_cloud"])
    )
    return long[keep]


def scenes_once(long: pd.DataFrame) -> pd.DataFrame:
    columns = [stats.SCENE_KEY, "site_no", "time_utc", "spacecraft"]
    return (
        long[columns]
        .drop_duplicates(stats.SCENE_KEY)
        .assign(year=lambda frame: frame["time_utc"].dt.year)
    )


def scene_counts(long: pd.DataFrame) -> pd.DataFrame:
    scenes = scenes_once(long)
    table = scenes.groupby(["site_no", "year", "spacecraft"]).size().unstack("spacecraft")
    table = table.fillna(0).astype(int)
    table["total"] = table.sum(axis=1)
    return table.reset_index()


def valid_pixel_loss(long: pd.DataFrame) -> pd.DataFrame:
    grouped = long.groupby(["roi", "erosion_px", "cloud_variant"])
    summary = grouped[["valid_count", "total_count"]].sum()
    summary["pixel_loss"] = 1.0 - summary["valid_count"] / summary["total_count"]
    summary["median_scene_valid_fraction"] = grouped["valid_fraction"].median()
    return summary.reset_index()


def zero_valid_site_years(
    long: pd.DataFrame, sites: list[str], years: list[int], settings: dict
) -> pd.DataFrame:
    reference = reference_rows(long, settings).assign(year=lambda frame: frame["time_utc"].dt.year)
    usable = reference[reference["valid_count"] > 0].groupby(["site_no", "year"]).size()
    present = reference.groupby(["site_no", "year"]).size()
    rows = []
    for site in sites:
        for year in years:
            scenes = int(present.get((site, year), 0))
            valid = int(usable.get((site, year), 0))
            if valid == 0:
                rows.append({"site_no": site, "year": year, "scenes": scenes, "valid_scenes": 0})
    return pd.DataFrame(rows, columns=["site_no", "year", "scenes", "valid_scenes"])


def relative_difference(first: pd.Series, second: pd.Series) -> float:
    centre = float(second.median())
    if centre == 0 or np.isnan(centre):
        return float("nan")
    return float((first.median() - centre) / abs(centre))


def sensor_jump(long: pd.DataFrame, settings: dict, bands: list[str]) -> pd.DataFrame:
    reference = reference_rows(long, settings)
    reference = reference[reference["valid_count"] > 0]
    split = pd.Timestamp(year=settings["split_year"], month=1, day=1, tz="UTC")
    minimum = settings["minimum_scenes_per_group"]
    rows = []
    for site in settings["sites"]:
        subset = reference[reference["site_no"] == site]
        before = subset[subset["time_utc"] < split]
        after = subset[subset["time_utc"] >= split]
        late_c = after[after["spacecraft"] == "S2C"]
        late_ab = after[after["spacecraft"].isin(["S2A", "S2B"])]
        for band in bands:
            column = f"{band}_median"
            for label, left, right in (
                ("after_vs_before", after, before),
                ("S2C_vs_S2A_S2B_after_split", late_c, late_ab),
            ):
                enough = len(left) >= minimum and len(right) >= minimum
                difference = relative_difference(left[column], right[column]) if enough else np.nan
                rows.append(
                    {
                        "site_no": site,
                        "band": band,
                        "comparison": label,
                        "n_left": len(left),
                        "n_right": len(right),
                        "relative_difference": difference,
                        "flag": bool(abs(difference) > settings["jump_threshold_relative"])
                        if not np.isnan(difference)
                        else False,
                    }
                )
    return pd.DataFrame(rows)


def month_matched_difference(
    left: pd.DataFrame, right: pd.DataFrame, column: str, minimum: int
) -> tuple[float, int]:
    differences = []
    for month in range(1, 13):
        first = left[left["time_utc"].dt.month == month][column]
        second = right[right["time_utc"].dt.month == month][column]
        if len(first) >= minimum and len(second) >= minimum:
            differences.append(relative_difference(first, second))
    if not differences:
        return float("nan"), 0
    return float(np.nanmedian(differences)), len(differences)


def matched_comparisons(long: pd.DataFrame, settings: dict, bands: list[str]) -> pd.DataFrame:
    reference = reference_rows(long, settings)
    reference = reference[reference["valid_count"] > 0]
    split = pd.Timestamp(year=settings["split_year"], month=1, day=1, tz="UTC")
    minimum = settings["minimum_scenes_per_month"]
    rows = []
    for site in settings["sites"]:
        subset = reference[reference["site_no"] == site]
        before = subset[subset["time_utc"] < split]
        after = subset[subset["time_utc"] >= split]
        groups = {
            "S2C_vs_S2A_S2B_after_split": (
                after[after["spacecraft"] == "S2C"],
                after[after["spacecraft"].isin(["S2A", "S2B"])],
            ),
            "after_vs_before_S2A_S2B": (
                after[after["spacecraft"].isin(["S2A", "S2B"])],
                before[before["spacecraft"].isin(["S2A", "S2B"])],
            ),
            "S2A_vs_S2B_before_split": (
                before[before["spacecraft"] == "S2A"],
                before[before["spacecraft"] == "S2B"],
            ),
        }
        for band in bands:
            for label, (left, right) in groups.items():
                difference, months = month_matched_difference(
                    left, right, f"{band}_median", minimum
                )
                rows.append(
                    {
                        "site_no": site,
                        "band": band,
                        "comparison": label,
                        "months_compared": months,
                        "median_relative_difference": difference,
                    }
                )
    return pd.DataFrame(rows)
