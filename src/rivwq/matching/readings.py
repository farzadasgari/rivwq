from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rivwq.usgs import qc_rules

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "matching.yaml"
SECONDS_PER_MINUTE = 60.0
NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass
class ReadingSeries:
    site_no: str
    parameter_code: str
    sensor_id: str
    times_ns: np.ndarray
    values: np.ndarray
    provisional: np.ndarray
    estimated: np.ndarray
    native_by_year: dict[int, float]
    native_overall: float
    extra: dict = field(default_factory=dict)

    def native_minutes(self, year: int) -> float:
        value = self.native_by_year.get(year, float("nan"))
        return self.native_overall if np.isnan(value) else value


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def require_utc(times: pd.Series) -> None:
    if not pd.api.types.is_datetime64_any_dtype(times) or times.dt.tz is None:
        raise ValueError("timestamps must be timezone-aware")
    if str(times.dt.tz) != "UTC":
        raise ValueError("timestamps must be in UTC")


def to_ns(times: pd.Series) -> np.ndarray:
    require_utc(times)
    naive = times.dt.tz_localize(None).astype("datetime64[ns]")
    return naive.to_numpy().astype("int64")


def median_step_minutes(times: pd.Series) -> float:
    unique = times.drop_duplicates().sort_values()
    if len(unique) < 2:
        return float("nan")
    return float(unique.diff().dropna().dt.total_seconds().median() / SECONDS_PER_MINUTE)


def whole_minutes(value: float) -> float:
    return float("nan") if np.isnan(value) else float(round(value))


def native_interval_table(tidy: pd.DataFrame) -> pd.DataFrame:
    require_utc(tidy["datetime_utc"])
    observed = tidy[tidy["value"].notna()]
    rows = []
    keys = ["site_no", "parameter_code", "sensor_id"]
    for key, group in observed.groupby(keys, sort=True):
        years = group["datetime_utc"].dt.year
        for year, part in group.groupby(years, sort=True):
            rows.append(
                {
                    "site_no": key[0],
                    "parameter_code": key[1],
                    "sensor_id": key[2],
                    "year": int(year),
                    "native_minutes": whole_minutes(median_step_minutes(part["datetime_utc"])),
                    "values": len(part),
                }
            )
    return pd.DataFrame(rows, columns=[*keys, "year", "native_minutes", "values"])


def native_lookup(table: pd.DataFrame) -> dict[tuple, dict[int, float]]:
    result: dict[tuple, dict[int, float]] = {}
    for row in table.itertuples(index=False):
        key = (row.site_no, row.parameter_code, row.sensor_id)
        result.setdefault(key, {})[int(row.year)] = float(row.native_minutes)
    return result


def overall_native(years: dict[int, float], counts: dict[int, int]) -> float:
    valid = {year: value for year, value in years.items() if not np.isnan(value)}
    if not valid:
        return float("nan")
    weights = np.array([counts[year] for year in valid], dtype=float)
    ordered = np.array(list(valid.values()))
    order = np.argsort(ordered)
    cumulative = np.cumsum(weights[order])
    return float(ordered[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def build_series(
    tidy: pd.DataFrame, config: dict, qc_config: dict, level: str, parameter_code: str
) -> list[ReadingSeries]:
    selected = tidy[tidy["parameter_code"] == parameter_code]
    if selected.empty:
        return []
    natives = native_interval_table(selected)
    lookup = native_lookup(natives)
    flags = qc_rules.apply_level(selected, qc_config, level)
    kept = selected[~flags["exclude"].to_numpy()]
    kept = kept.drop_duplicates(["sensor_id", "datetime_utc", "value"])
    estimated_token = qc_rules.qualifier_pattern([config["estimated_qualifier"]])
    result = []
    for sensor_id, group in kept.groupby("sensor_id", sort=True):
        ordered = group.sort_values("datetime_utc", kind="stable")
        key = (ordered["site_no"].iloc[0], parameter_code, sensor_id)
        years = lookup.get(key, {})
        counts = natives[
            (natives["sensor_id"] == sensor_id) & (natives["parameter_code"] == parameter_code)
        ].set_index("year")["values"]
        result.append(
            ReadingSeries(
                site_no=key[0],
                parameter_code=parameter_code,
                sensor_id=sensor_id,
                times_ns=to_ns(ordered["datetime_utc"]),
                values=ordered["value"].to_numpy(dtype="float64"),
                provisional=(ordered["approval_status"] == config["provisional_status"]).to_numpy(),
                estimated=ordered["qualifiers"].fillna("").str.contains(estimated_token).to_numpy(),
                native_by_year=years,
                native_overall=overall_native(years, counts.to_dict()),
            )
        )
    return result
