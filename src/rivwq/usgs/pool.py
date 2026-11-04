from __future__ import annotations

import pandas as pd

MISSING_SERIES_REASON = "missing_required_series"


def parse_reasons(text: object) -> set[str]:
    if not isinstance(text, str) or not text:
        return set()
    return set(text.split(";"))


def required_variables(scout_config: dict) -> list[str]:
    return [name for name, spec in scout_config["variables"].items() if spec["required"]]


def fails_only_for_missing_series(row: pd.Series, variable: str, required: list[str]) -> bool:
    if parse_reasons(row["fail_reasons"]) != {MISSING_SERIES_REASON}:
        return False
    if pd.notna(row[f"{variable}_code"]):
        return False
    others = [name for name in required if name != variable]
    return all(pd.notna(row[f"{name}_code"]) for name in others)


def relaxed_sites(screened: pd.DataFrame, variable: str, required: list[str]) -> pd.Index:
    mask = screened.apply(
        lambda row: fails_only_for_missing_series(row, variable, required), axis=1
    )
    return screened.index[mask]


def build_pool(
    candidates: pd.DataFrame,
    screened: pd.DataFrame,
    done_sites: set[str],
    pool_config: dict,
    required: list[str],
) -> pd.DataFrame:
    minimum_area = pool_config["min_drainage_area_sqmi"]
    large_candidates = candidates.index[candidates["drainage_area_sqmi"] >= minimum_area]
    relaxed = relaxed_sites(screened, pool_config["relaxed_missing_series"], required)
    relaxed_large = relaxed[screened.loc[relaxed, "drainage_area_sqmi"] >= minimum_area]
    sites = large_candidates.union(relaxed_large)
    pool = pd.DataFrame(index=sites.rename("site_no"))
    pool["relaxed_pool"] = pool.index.isin(relaxed_large)
    pool["already_done"] = pool.index.isin(done_sites)
    return pool


def normalise_code(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.replace(".", "", 1).isdigit():
        text = str(int(float(text)))
    return text.zfill(5)
