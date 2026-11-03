from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "scout.yaml"
EARTH_RADIUS_KM = 6371.0
DAYS_PER_YEAR = 365.25
TEXT_COLUMNS = {"site_no": str, "parm_cd": str, "huc_cd": str, "parameter_code": str}


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_codes(parameter_table: pd.DataFrame, spec: dict) -> pd.DataFrame:
    names = parameter_table["parameter_name"].fillna("")
    mask = names.str.contains(spec["name_pattern"], regex=True)
    if spec.get("exclude_pattern"):
        mask &= ~names.str.contains(spec["exclude_pattern"], regex=True)
    if spec.get("units"):
        mask &= parameter_table["unit_of_measure"].isin(spec["units"])
    columns = ["parameter_code", "parameter_name", "unit_of_measure"]
    return parameter_table.loc[mask, columns].reset_index(drop=True)


def resolve_all_codes(parameter_table: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    return {
        name: resolve_codes(parameter_table, spec) for name, spec in config["variables"].items()
    }


def tidal_codes(parameter_table: pd.DataFrame, config: dict) -> set[str]:
    pattern = config["criteria"]["tidal_parameter_pattern"]
    names = parameter_table["parameter_name"].fillna("")
    return set(parameter_table.loc[names.str.contains(pattern, regex=True), "parameter_code"])


def instantaneous_series(catalog: pd.DataFrame) -> pd.DataFrame:
    series = catalog[(catalog["data_type_cd"] == "uv") & catalog["parm_cd"].notna()].copy()
    series["begin_date"] = pd.to_datetime(series["begin_date"], errors="coerce")
    series["end_date"] = pd.to_datetime(series["end_date"], errors="coerce")
    series["count_nu"] = pd.to_numeric(series["count_nu"], errors="coerce")
    return series.dropna(subset=["begin_date", "end_date"])


def summarise_variable(series: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    selected = series[series["parm_cd"].isin(codes)]
    per_code = selected.groupby(["site_no", "parm_cd"], as_index=False).agg(
        first=("begin_date", "min"), last=("end_date", "max"), days=("count_nu", "sum")
    )
    per_code = per_code.sort_values(["site_no", "days", "parm_cd"], ascending=[True, False, True])
    best = per_code.drop_duplicates("site_no").set_index("site_no")
    span_days = (best["last"] - best["first"]).dt.days + 1
    result = pd.DataFrame(
        {
            "code": best["parm_cd"],
            "first": best["first"],
            "last": best["last"],
            "catalog_day_fraction": (best["days"] / span_days).clip(upper=1.0),
        }
    )
    return result


def wide_series_table(series: pd.DataFrame, code_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sites = pd.DataFrame(index=pd.Index(sorted(series["site_no"].unique()), name="site_no"))
    for variable, codes in code_map.items():
        summary = summarise_variable(series, list(codes["parameter_code"]))
        summary.columns = [f"{variable}_{column}" for column in summary.columns]
        sites = sites.join(summary)
    return sites


def apply_criteria(
    table: pd.DataFrame, config: dict, tidal_sites: set[str], reference_sites: set[str]
) -> pd.DataFrame:
    criteria = config["criteria"]
    required = [name for name, spec in config["variables"].items() if spec["required"]]
    result = table.copy()
    firsts = result[[f"{name}_first" for name in required]]
    lasts = result[[f"{name}_last" for name in required]]
    fractions = result[[f"{name}_catalog_day_fraction" for name in required]]
    sentinel_start = pd.Timestamp(config["sentinel2_start"])
    result["overlap_start"] = firsts.max(axis=1).clip(lower=sentinel_start)
    result["overlap_end"] = lasts.min(axis=1)
    overlap_days = (result["overlap_end"] - result["overlap_start"]).dt.days
    result["overlap_years"] = (overlap_days / DAYS_PER_YEAR).clip(lower=0.0)
    result["min_required_catalog_day_fraction"] = fractions.min(axis=1)
    result["fluorescence_present"] = result["chlorophyll_fluorescence_code"].notna()
    result["is_reference"] = result.index.isin(reference_sites)
    result["is_tidal"] = result.index.isin(tidal_sites)

    failures = pd.DataFrame(index=result.index)
    failures["missing_required_series"] = firsts.isna().any(axis=1)
    failures["not_stream_site"] = result["site_tp_cd"] != config["site_type"]
    failures["tidal_or_canal"] = result["is_tidal"] | result["station_nm"].str.contains(
        criteria["excluded_name_pattern"], case=False, regex=True
    )
    failures["overlap_too_short"] = result["overlap_years"] < criteria["min_overlap_years"]
    failures["too_many_gaps"] = (
        result["min_required_catalog_day_fraction"] < criteria["min_day_fraction"]
    )
    failures["record_ended_early"] = result["overlap_end"] < pd.Timestamp(criteria["min_last_date"])
    result["fail_reasons"] = failures.apply(
        lambda row: ";".join(name for name, failed in row.items() if failed), axis=1
    )
    result["meets_criteria"] = result["fail_reasons"] == ""
    return result


def funnel(table: pd.DataFrame) -> pd.Series:
    reasons = table["fail_reasons"].str.split(";").explode()
    counts = reasons[reasons != ""].value_counts()
    counts["screened_sites"] = len(table)
    counts["meets_criteria"] = int(table["meets_criteria"].sum())
    return counts


def median_of_daily_medians(daily_stats: pd.DataFrame) -> float:
    if daily_stats.empty:
        return np.nan
    return float(pd.to_numeric(daily_stats["p50_va"], errors="coerce").median())


def value_columns(frame: pd.DataFrame, code: str) -> list[str]:
    return [
        column
        for column in frame.columns
        if (column == code or column.startswith(f"{code}_")) and not column.endswith("_cd")
    ]


def median_of_instantaneous_values(frames: list[pd.DataFrame], code: str) -> float:
    values = [
        pd.to_numeric(frame[column], errors="coerce").dropna()
        for frame in frames
        for column in value_columns(frame, code)
    ]
    if not values:
        return np.nan
    return float(pd.concat(values).median())


def day_fraction_from_instantaneous(frame: pd.DataFrame, code: str, start: str, end: str) -> float:
    window_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    if frame is None:
        return 0.0
    numeric = frame[value_columns(frame, code)].apply(pd.to_numeric, errors="coerce")
    present = numeric.dropna(how="all")
    if present.empty:
        return 0.0
    index = pd.DatetimeIndex(present.index)
    if index.tz is not None:
        index = index.tz_convert("UTC")
    return min(index.normalize().nunique() / window_days, 1.0)


def seasonal_peak_month(monthly_stats: pd.DataFrame) -> float:
    if monthly_stats.empty:
        return np.nan
    values = pd.to_numeric(monthly_stats["mean_va"], errors="coerce")
    by_month = values.groupby(monthly_stats["month_nu"].astype(int)).mean()
    return float(by_month.idxmax())


def regulated_peak_share(peaks: pd.DataFrame, regulation_codes: list[str]) -> float:
    if peaks.empty:
        return np.nan
    flagged = (
        peaks["peak_cd"]
        .fillna("")
        .apply(lambda value: any(code in str(value).split(",") for code in regulation_codes))
    )
    return float(flagged.mean())


def classify_turbidity(median_fnu: float, strata: dict) -> str:
    if np.isnan(median_fnu):
        return "unknown"
    if median_fnu >= strata["turbid_median_fnu_min"]:
        return "turbid"
    if median_fnu <= strata["clear_median_fnu_max"]:
        return "clear"
    return "intermediate"


def classify_regulation(share: float, strata: dict) -> str:
    if np.isnan(share):
        return "unknown"
    if share >= strata["regulated_peak_share_min"]:
        return "regulated"
    if share <= strata["free_flowing_peak_share_max"]:
        return "free_flowing"
    return "mixed"


def classify_regime(peak_month: float, altitude_ft: float, latitude: float, strata: dict) -> str:
    if np.isnan(peak_month):
        return "unknown"
    high = altitude_ft >= strata["snowmelt_min_altitude_ft"]
    if high and peak_month in strata["snowmelt_peak_months"]:
        return "snowmelt"
    north = latitude >= strata["northern_snowmelt_min_latitude"]
    if north and peak_month in strata["northern_snowmelt_peak_months"]:
        return "snowmelt"
    return "rain_fed"


def classify_cloudiness(state: str, strata: dict) -> str:
    if state in strata["cloudier_states"]:
        return "cloudier"
    if state in strata["sunnier_states"]:
        return "sunnier"
    return "unknown"


def add_strata(table: pd.DataFrame, summaries: pd.DataFrame, strata: dict) -> pd.DataFrame:
    result = table.join(summaries)
    result["turbidity_class"] = result["turbidity_median_fnu"].apply(
        lambda value: classify_turbidity(value, strata)
    )
    result["regulation_class"] = result["regulated_peak_share"].apply(
        lambda value: classify_regulation(value, strata)
    )
    result["regime_class"] = [
        classify_regime(peak, altitude, latitude, strata)
        for peak, altitude, latitude in zip(
            result["discharge_peak_month"], result["alt_va"], result["dec_lat_va"], strict=True
        )
    ]
    result["cloud_class"] = result["state"].apply(lambda state: classify_cloudiness(state, strata))
    return result


STRATUM_AXES = ["turbidity_class", "regulation_class", "regime_class", "cloud_class"]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def quality_score(row: pd.Series, settings: dict) -> float:
    length = min(row["overlap_years"] / settings["max_overlap_years_for_quality"], 1.0)
    return float(row["sampled_day_fraction"] * length)


def select_shortlist(candidates: pd.DataFrame, settings: dict) -> pd.DataFrame:
    eligible = candidates["meets_criteria"] & ~candidates["is_reference"]
    eligible &= candidates["drainage_area_sqmi"] >= settings["min_drainage_area_sqmi"]
    eligible &= candidates["sampled_day_fraction"] >= settings["min_sampled_day_fraction"]
    for axis in STRATUM_AXES:
        eligible &= candidates[axis] != "unknown"
    pool = candidates[eligible].copy()
    pool["quality"] = pool.apply(lambda row: quality_score(row, settings), axis=1)
    counts = {axis: {} for axis in STRATUM_AXES}
    chosen: list[str] = []
    while len(chosen) < settings["size"] and not pool.empty:
        gains = pd.Series(0.0, index=pool.index)
        for axis in STRATUM_AXES:
            seen = pool[axis].map(lambda value, axis=axis: counts[axis].get(value, 0))
            known = pool[axis] != "unknown"
            gains += np.where(known, 1.0 / (1.0 + seen), 0.0)
        gains += settings["quality_weight"] * pool["quality"]
        gains += settings["fluorescence_bonus"] * pool["fluorescence_present"].astype(float)
        for site in chosen:
            near = pool.apply(
                lambda row, site=site: (
                    haversine_km(
                        row["dec_lat_va"],
                        row["dec_long_va"],
                        candidates.at[site, "dec_lat_va"],
                        candidates.at[site, "dec_long_va"],
                    )
                    < settings["min_separation_km"]
                ),
                axis=1,
            )
            gains = gains.where(~near, -np.inf)
        ordered = gains.sort_values(ascending=False, kind="stable")
        ordered = ordered[np.isfinite(ordered)]
        if ordered.empty:
            break
        best = sorted(ordered[ordered == ordered.iloc[0]].index)[0]
        chosen.append(best)
        for axis in STRATUM_AXES:
            value = pool.at[best, axis]
            counts[axis][value] = counts[axis].get(value, 0) + 1
        pool = pool.drop(index=best)
    shortlist = candidates.loc[chosen].copy()
    shortlist["reason"] = shortlist.apply(describe_pick, axis=1)
    return shortlist


def describe_pick(row: pd.Series) -> str:
    extras = " with fChl" if row["fluorescence_present"] else ""
    return (
        f"{row['turbidity_class']}, {row['regulation_class'].replace('_', ' ')}, "
        f"{row['regime_class'].replace('_', ' ')}, {row['cloud_class']} region; "
        f"{row['overlap_years']:.1f} y overlap, {row['sampled_day_fraction']:.0%} of 2025 days"
        f"{extras}"
    )


def chunked(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def cached(path: Path, refresh: bool, build) -> pd.DataFrame:
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype=TEXT_COLUMNS)
    frame = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def fetch_parameter_table() -> pd.DataFrame:
    from dataretrieval import waterdata

    table, _ = waterdata.get_reference_table("parameter-codes")
    return table


def fetch_state_catalog(state: str, codes: list[str], site_type: str) -> pd.DataFrame:
    from dataretrieval import nwis

    catalog, _ = nwis.get_info(
        stateCd=state,
        siteType=site_type,
        parameterCd=",".join(codes),
        seriesCatalogOutput=True,
    )
    catalog = catalog.drop(columns="geometry", errors="ignore")
    catalog["queried_state"] = state
    return catalog


def fetch_site_details(site_numbers: list[str], batch_size: int, pause: float) -> pd.DataFrame:
    from dataretrieval import nwis

    frames = []
    for batch in chunked(site_numbers, batch_size):
        details, _ = nwis.get_info(sites=",".join(batch), siteOutput="expanded")
        frames.append(details.drop(columns="geometry", errors="ignore"))
        time.sleep(pause)
    return pd.concat(frames, ignore_index=True)


def with_retries(call, attempts: int, pause: float):
    for attempt in range(attempts):
        try:
            return call()
        except Exception:
            if attempt == attempts - 1:
                return None
            time.sleep(pause * 2 ** (attempt + 1))
    return None


def fetch_one_site_summary(site: str, row: pd.Series, config: dict) -> dict:
    from dataretrieval import nwis

    pause = config["request_pause_seconds"]
    attempts = config["retry_attempts"]
    result = {
        "site_no": site,
        "turbidity_median_fnu": np.nan,
        "turbidity_median_source": "nwis_daily_statistics",
        "discharge_peak_month": np.nan,
    }
    if pd.notna(row["turbidity_code"]):
        daily = with_retries(
            lambda: nwis.get_stats(
                sites=site,
                parameterCd=row["turbidity_code"],
                statReportType="daily",
                statType="all",
            )[0],
            attempts,
            pause,
        )
        if daily is not None:
            result["turbidity_median_fnu"] = median_of_daily_medians(daily)
        if np.isnan(result["turbidity_median_fnu"]):
            windows = []
            for start, end in config["fallback_windows"]:
                frame = with_retries(
                    lambda start=start, end=end: nwis.get_iv(
                        sites=site, parameterCd=row["turbidity_code"], start=start, end=end
                    )[0],
                    attempts,
                    pause,
                )
                if frame is not None:
                    windows.append(frame)
            result["turbidity_median_fnu"] = median_of_instantaneous_values(
                windows, row["turbidity_code"]
            )
            result["turbidity_median_source"] = "sampled_iv_windows"
        time.sleep(pause)
    if pd.notna(row["discharge_code"]):
        monthly = with_retries(
            lambda: nwis.get_stats(
                sites=site,
                parameterCd=row["discharge_code"],
                statReportType="monthly",
                statType="mean",
            )[0],
            attempts,
            pause,
        )
        if monthly is not None:
            result["discharge_peak_month"] = seasonal_peak_month(monthly)
        time.sleep(pause)
    return result


def fetch_site_summaries(table: pd.DataFrame, config: dict, partial_path: Path) -> pd.DataFrame:
    from dataretrieval import nwis

    pause = config["request_pause_seconds"]
    attempts = config["retry_attempts"]
    codes = config["strata"]["regulation_peak_codes"]
    done = pd.read_csv(partial_path, dtype=TEXT_COLUMNS) if partial_path.exists() else None
    if done is not None:
        incomplete = done["turbidity_median_fnu"].isna() | done["discharge_peak_month"].isna()
        done = done[~incomplete]
    rows = [] if done is None else done.to_dict("records")
    finished = set() if done is None else set(done["site_no"])
    for site, row in table.iterrows():
        if site in finished:
            continue
        rows.append(fetch_one_site_summary(site, row, config))
        pd.DataFrame(rows).to_csv(partial_path, index=False)
    summaries = pd.DataFrame(rows).set_index("site_no")
    peak_shares = {}
    for batch in chunked(list(table.index), config["batch_size"]):
        peaks = with_retries(
            lambda batch=batch: nwis.get_discharge_peaks(sites=batch, multi_index=False)[0],
            attempts,
            pause,
        )
        if peaks is not None:
            for site, site_peaks in peaks.groupby("site_no"):
                peak_shares[site] = regulated_peak_share(site_peaks, codes)
        time.sleep(pause)
    summaries["regulated_peak_share"] = pd.Series(peak_shares)
    return summaries.reset_index()


def fetch_coverage(table: pd.DataFrame, config: dict, partial_path: Path) -> pd.DataFrame:
    from dataretrieval import nwis

    start, end = config["coverage_window"]
    pause = config["request_pause_seconds"]
    attempts = config["retry_attempts"]
    rows = []
    finished = set()
    if partial_path.exists():
        done = pd.read_csv(partial_path, dtype=TEXT_COLUMNS)
        done = done.dropna(subset=["sampled_day_fraction"])
        rows = done.to_dict("records")
        finished = set(done["site_no"])
    for site, row in table.iterrows():
        if site in finished:
            continue
        code = str(row["turbidity_code"])
        frame = with_retries(
            lambda site=site, code=code: nwis.get_iv(
                sites=site, parameterCd=code, start=start, end=end
            )[0],
            attempts,
            pause,
        )
        fraction = (
            np.nan if frame is None else day_fraction_from_instantaneous(frame, code, start, end)
        )
        rows.append({"site_no": site, "sampled_day_fraction": fraction})
        pd.DataFrame(rows).to_csv(partial_path, index=False)
        time.sleep(pause)
    return pd.DataFrame(rows)


def collect(config: dict, refresh: bool) -> dict[str, pd.DataFrame]:
    cache = REPO_ROOT / config["cache_dir"]
    pause = config["request_pause_seconds"]
    parameters = cached(cache / "parameter_codes.csv", refresh, fetch_parameter_table)
    code_map = resolve_all_codes(parameters, config)
    turbidity_codes = list(code_map["turbidity"]["parameter_code"])

    def catalog_all_states() -> pd.DataFrame:
        frames = []
        for state in config["states"]:
            frames.append(fetch_state_catalog(state, turbidity_codes, config["site_type"]))
            time.sleep(pause)
        return pd.concat(frames, ignore_index=True)

    catalog = cached(cache / "series_catalog.csv", refresh, catalog_all_states)
    return {"parameters": parameters, "catalog": catalog}


def build_tables(
    config: dict, parameters: pd.DataFrame, catalog: pd.DataFrame, details: pd.DataFrame
) -> pd.DataFrame:
    code_map = resolve_all_codes(parameters, config)
    series = instantaneous_series(catalog)
    wide = wide_series_table(series, code_map)
    site_state = catalog.drop_duplicates("site_no").set_index("site_no")["queried_state"]
    site_columns = ["station_nm", "site_tp_cd", "dec_lat_va", "dec_long_va"]
    site_columns += ["huc_cd", "alt_va", "drain_area_va"]
    info = details.drop_duplicates("site_no").set_index("site_no")[site_columns]
    table = info.join(wide, how="inner").join(site_state.rename("state"))
    table["huc8"] = table["huc_cd"].str[:8]
    table = table.rename(columns={"drain_area_va": "drainage_area_sqmi"})
    all_tidal = tidal_codes(parameters, config)
    tidal_sites = set(catalog.loc[catalog["parm_cd"].isin(all_tidal), "site_no"])
    return apply_criteria(table, config, tidal_sites, set(config["reference_sites"]))


def write_outputs(config: dict, candidates: pd.DataFrame, shortlist: pd.DataFrame) -> list[Path]:
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = config["output_prefix"]
    keep = candidates["meets_criteria"] | candidates["is_reference"]
    paths = {
        output_dir / f"{prefix}_candidates.csv": candidates[keep],
        output_dir / f"{prefix}_screened_all.csv": candidates,
        output_dir / f"{prefix}_shortlist.csv": shortlist,
    }
    for path, frame in paths.items():
        frame.to_csv(path, index_label="site_no")
    return list(paths)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    cache = REPO_ROOT / config["cache_dir"]
    pause = config["request_pause_seconds"]
    collected = collect(config, args.refresh)
    parameters, catalog = collected["parameters"], collected["catalog"]
    missing = [site for site in config["reference_sites"] if site not in set(catalog["site_no"])]
    if missing:
        raise SystemExit(f"reference sites missing from catalog: {missing}")
    sites = sorted(catalog["site_no"].unique())
    details = cached(
        cache / "site_details.csv",
        args.refresh,
        lambda: fetch_site_details(sites, config["batch_size"], pause),
    )
    table = build_tables(config, parameters, catalog, details)
    print(funnel(table).to_string())
    selected = table[table["meets_criteria"] | table["is_reference"]]
    summaries = cached(
        cache / "site_summaries.csv",
        args.refresh,
        lambda: fetch_site_summaries(selected, config, cache / "site_summaries_partial.csv"),
    ).set_index("site_no")
    coverage = cached(
        cache / "site_coverage.csv",
        args.refresh,
        lambda: fetch_coverage(selected, config, cache / "site_coverage_partial.csv"),
    ).set_index("site_no")
    candidates = add_strata(table, summaries.join(coverage), config["strata"])
    shortlist = select_shortlist(candidates, config["shortlist"])
    for path in write_outputs(config, candidates, shortlist):
        print(path)
