from __future__ import annotations

import re

import numpy as np
import pandas as pd

TIDY_COLUMNS = [
    "site_no",
    "datetime_utc",
    "parameter_code",
    "sensor_id",
    "value",
    "qualifiers",
    "approval_status",
    "unit_of_measure",
    "source_column",
]
OFFSET_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
QUALIFIER_SEPARATORS = re.compile(r"[,;|\s]+")
QUALIFIER_JOINER = "|"
SITE_PREFIX = "USGS-"


def to_utc(values: pd.Series) -> pd.Series:
    if values.isna().any():
        raise ValueError("timestamps contain missing values")
    if pd.api.types.is_datetime64_any_dtype(values):
        if values.dt.tz is None:
            raise ValueError("timestamps carry no time zone and cannot be converted to UTC")
        return values.dt.tz_convert("UTC").astype("datetime64[us, UTC]")
    text = values.astype(str)
    if not text.str.contains(OFFSET_SUFFIX).all():
        raise ValueError("timestamp strings must end with Z or a UTC offset")
    return pd.to_datetime(text, utc=True, format="ISO8601").astype("datetime64[us, UTC]")


def parse_qualifiers(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return QUALIFIER_JOINER.join(parts)
    text = str(value).strip().strip("{}[]()'\"")
    parts = [part.strip("'\"") for part in QUALIFIER_SEPARATORS.split(text) if part]
    return QUALIFIER_JOINER.join(parts)


def site_number(monitoring_location_id: pd.Series) -> pd.Series:
    return monitoring_location_id.str.removeprefix(SITE_PREFIX)


def tidy_response(response: pd.DataFrame, source_column: str) -> pd.DataFrame:
    tidy = pd.DataFrame(
        {
            "site_no": site_number(response["monitoring_location_id"]).astype(str),
            "datetime_utc": to_utc(response["time"]),
            "parameter_code": response["parameter_code"].astype(str),
            "sensor_id": response["time_series_id"].astype(str),
            "value": pd.to_numeric(response["value"], errors="raise").astype("float64"),
            "qualifiers": response["qualifier"].map(parse_qualifiers).astype(str),
            "approval_status": response["approval_status"].astype(str),
            "unit_of_measure": response["unit_of_measure"].astype(str),
            "source_column": source_column,
        }
    )
    ordered = tidy.sort_values(["parameter_code", "sensor_id", "datetime_utc"], kind="stable")
    return ordered.reset_index(drop=True)[TIDY_COLUMNS]
