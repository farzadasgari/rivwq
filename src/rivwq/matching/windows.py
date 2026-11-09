from __future__ import annotations

import numpy as np
import pandas as pd

from rivwq.matching.readings import ReadingSeries, to_ns

NEAREST = "nearest"
DAILY = "daily_mean"
RESULT_COLUMNS = [
    "scene_pos",
    "variant",
    "sensor_id",
    "value",
    "n",
    "dt_min",
    "provisional_share",
    "estimated_share",
    "native_min",
    "other_sensor_in_window",
]


def window_variants(config: dict) -> list[str]:
    return [NEAREST] + [f"median_{minutes}" for minutes in config["windows_minutes"]] + [DAILY]


def nearest_in_slice(times: np.ndarray, target: int) -> int:
    distance = np.abs(times - target)
    return int(np.argmin(distance))


def summarise(
    series: ReadingSeries, lo: int, hi: int, target: int, statistic: str, per_minute: int
) -> dict:
    count = hi - lo
    if count == 0:
        return {
            "value": np.nan,
            "n": 0,
            "dt_min": np.nan,
            "provisional_share": np.nan,
            "estimated_share": np.nan,
        }
    values = series.values[lo:hi]
    times = series.times_ns[lo:hi]
    nearest = nearest_in_slice(times, target)
    value = float(np.median(values)) if statistic == "median" else float(np.mean(values))
    return {
        "value": value,
        "n": count,
        "dt_min": float((times[nearest] - target) / per_minute),
        "provisional_share": float(series.provisional[lo:hi].mean()),
        "estimated_share": float(series.estimated[lo:hi].mean()),
    }


def nearest_bounds(series: ReadingSeries, target: int, tolerance_ns: float) -> tuple[int, int]:
    times = series.times_ns
    if times.size == 0:
        return 0, 0
    position = int(np.searchsorted(times, target, side="left"))
    candidates = [index for index in (position - 1, position) if 0 <= index < times.size]
    distances = [abs(int(times[index]) - target) for index in candidates]
    best = candidates[int(np.argmin(distances))]
    if abs(int(times[best]) - target) > tolerance_ns:
        return 0, 0
    return best, best + 1


def match_series(series: ReadingSeries, scene_times: pd.Series, config: dict) -> pd.DataFrame:
    per_minute = config["nanoseconds_per_minute"]
    day_ns = config["minutes_per_day"] * per_minute
    targets = to_ns(scene_times)
    years = scene_times.dt.year.to_numpy()
    rows = []
    for position, (target, year) in enumerate(zip(targets, years, strict=True)):
        native = series.native_minutes(int(year))
        tolerance = config["nearest_max_fraction_of_native"] * native * per_minute
        bounds = {NEAREST: nearest_bounds(series, int(target), tolerance)}
        for minutes in config["windows_minutes"]:
            half = minutes * per_minute
            bounds[f"median_{minutes}"] = (
                int(np.searchsorted(series.times_ns, target - half, side="left")),
                int(np.searchsorted(series.times_ns, target + half, side="right")),
            )
        day_start = (int(target) // day_ns) * day_ns
        bounds[DAILY] = (
            int(np.searchsorted(series.times_ns, day_start, side="left")),
            int(np.searchsorted(series.times_ns, day_start + day_ns, side="left")),
        )
        for variant, (lo, hi) in bounds.items():
            statistic = "mean" if variant == DAILY else "median"
            row = summarise(series, lo, hi, int(target), statistic, per_minute)
            row.update(
                {
                    "scene_pos": position,
                    "variant": variant,
                    "sensor_id": series.sensor_id,
                    "native_min": native,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def choose_sensor(candidates: pd.DataFrame) -> pd.DataFrame:
    ranked = candidates.assign(
        _abs_dt=candidates["dt_min"].abs().fillna(np.inf), _neg_n=-candidates["n"]
    ).sort_values(["scene_pos", "variant", "_neg_n", "_abs_dt", "sensor_id"], kind="stable")
    with_data = candidates[candidates["n"] > 0].groupby(["scene_pos", "variant"]).size()
    chosen = ranked.drop_duplicates(["scene_pos", "variant"], keep="first").copy()
    keys = pd.MultiIndex.from_frame(chosen[["scene_pos", "variant"]])
    sensors_with_data = with_data.reindex(keys).fillna(0).to_numpy()
    chosen["other_sensor_in_window"] = sensors_with_data > 1
    chosen.loc[chosen["n"] == 0, "sensor_id"] = None
    return chosen.drop(columns=["_abs_dt", "_neg_n"])


def match_parameter(
    series_list: list[ReadingSeries], scene_times: pd.Series, config: dict
) -> pd.DataFrame:
    if not series_list:
        variants = window_variants(config)
        positions = np.repeat(np.arange(len(scene_times)), len(variants))
        return pd.DataFrame(
            {
                "scene_pos": positions,
                "variant": np.tile(variants, len(scene_times)),
                "sensor_id": None,
                "value": np.nan,
                "n": 0,
                "dt_min": np.nan,
                "provisional_share": np.nan,
                "estimated_share": np.nan,
                "native_min": np.nan,
                "other_sensor_in_window": False,
            }
        )[RESULT_COLUMNS]
    parts = [match_series(series, scene_times, config) for series in series_list]
    chosen = choose_sensor(pd.concat(parts, ignore_index=True))
    return chosen.sort_values(["scene_pos", "variant"]).reset_index(drop=True)[RESULT_COLUMNS]
