from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rivwq.gee import roi

CONFIG_PATH = roi.REPO_ROOT / "config" / "roi_connectivity.yaml"
CURRENT = "current"
RADIUS_KEYS = ("near_gauge", "reach", "extended")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def variant_name(kind: str, k: int) -> str:
    return f"{kind}_k{k}"


def variant_list(settings: dict) -> list[str]:
    names = [CURRENT]
    names += [variant_name("A", k) for k in settings["gap_join_pixels"]]
    names += [variant_name("B", k) for k in settings["gap_join_pixels"]]
    return names


def wide_table(long: pd.DataFrame, settings: dict) -> pd.DataFrame:
    table = long.pivot_table(index=["variant", "site_no"], columns="roi", values="area_km2")
    table = table[list(RADIUS_KEYS)].rename(
        columns={"near_gauge": "area_1km", "reach": "area_5km", "extended": "area_20km"}
    )
    table["ratio_20_5"] = table["area_20km"] / table["area_5km"].where(table["area_5km"] > 0)
    current = table.xs(CURRENT, level="variant")
    for column in ("area_1km", "area_5km", "area_20km"):
        baseline = current[column].reindex(table.index.get_level_values("site_no")).to_numpy()
        table[f"{column}_vs_current"] = table[column].to_numpy() / np.where(
            baseline > 0, baseline, np.nan
        )
    limits = settings["thresholds"]
    table["flag_broken"] = table["ratio_20_5"] < limits["broken_ratio"]
    table["flag_leakage"] = table["ratio_20_5"] > limits["leakage_ratio"]
    table["flag_tiny"] = table["area_1km"] < limits["tiny_area_km2"]
    return table.reset_index()


def flagged_sites(table: pd.DataFrame, variant: str, flag: str) -> set[str]:
    rows = table[(table["variant"] == variant) & table[flag]]
    return set(rows["site_no"])


def broken_sites(table: pd.DataFrame) -> set[str]:
    return flagged_sites(table, CURRENT, "flag_broken")


def choose_k(table: pd.DataFrame, settings: dict) -> dict:
    broken = broken_sites(table)
    leaking_now = flagged_sites(table, CURRENT, "flag_leakage")
    minimum = settings["selection"]["min_fixed_sites"]
    report = []
    chosen = None
    for k in settings["gap_join_pixels"]:
        name = variant_name("A", k)
        still_broken = flagged_sites(table, name, "flag_broken")
        fixed = broken - still_broken
        new_leaks = flagged_sites(table, name, "flag_leakage") - leaking_now
        qualifies = len(fixed) >= minimum and not new_leaks
        report.append(
            {
                "k": k,
                "fixed": sorted(fixed),
                "still_broken": sorted(broken & still_broken),
                "new_leakage": sorted(new_leaks),
                "qualifies": qualifies,
            }
        )
        if qualifies and chosen is None:
            chosen = k
    return {"chosen_k": chosen, "broken_now": sorted(broken), "candidates": report}


def decide_cap(table: pd.DataFrame, settings: dict, k: int | None) -> dict:
    if k is None:
        return {"adopt": False, "reason": "no k chosen"}
    base = variant_name("A", k)
    capped = variant_name("B", k)
    leaks_base = flagged_sites(table, base, "flag_leakage")
    leaks_cap = flagged_sites(table, capped, "flag_leakage")
    new_broken = flagged_sites(table, capped, "flag_broken") - flagged_sites(
        table, base, "flag_broken"
    )
    left = table[table["variant"] == base].set_index("site_no")["area_20km"]
    right = table[table["variant"] == capped].set_index("site_no")["area_20km"]
    retention = (right / left.where(left > 0)).dropna()
    retained = settings["selection"]["cap_min_area_retention"]
    lost = sorted(
        site for site, value in retention.items() if value < retained and site not in leaks_base
    )
    adopt = len(leaks_cap) < len(leaks_base) and not new_broken and not lost
    return {
        "adopt": adopt,
        "leakage_before": sorted(leaks_base),
        "leakage_after": sorted(leaks_cap),
        "new_broken": sorted(new_broken),
        "non_leaking_sites_losing_area": lost,
    }


def twin_component(water: np.ndarray, seed: np.ndarray, k: int) -> np.ndarray:
    closed = water.astype(bool)
    if k > 0:
        closed = erode_square(dilate_square(closed, k), k)
    visited = np.zeros_like(closed, dtype=bool)
    queue = deque(zip(*np.nonzero(seed & closed), strict=True))
    for cell in queue:
        visited[cell] = True
    rows, columns = closed.shape
    while queue:
        row, column = queue.popleft()
        for step_row, step_column in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row, next_column = row + step_row, column + step_column
            inside = 0 <= next_row < rows and 0 <= next_column < columns
            if inside and closed[next_row, next_column] and not visited[next_row, next_column]:
                visited[next_row, next_column] = True
                queue.append((next_row, next_column))
    return visited & water.astype(bool)


def dilate_square(mask: np.ndarray, k: int) -> np.ndarray:
    padded = np.pad(mask, k, constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    rows, columns = mask.shape
    for shift_row in range(2 * k + 1):
        for shift_column in range(2 * k + 1):
            result |= padded[shift_row : shift_row + rows, shift_column : shift_column + columns]
    return result


def erode_square(mask: np.ndarray, k: int) -> np.ndarray:
    return ~dilate_square(~mask, k)
