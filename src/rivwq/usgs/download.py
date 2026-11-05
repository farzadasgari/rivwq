from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import dataretrieval
import pandas as pd
import pyarrow.parquet as pq
import yaml
from dataretrieval import waterdata

from rivwq.usgs.tidy import SITE_PREFIX, TIDY_COLUMNS, tidy_response

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "usgs_download.yaml"
SERIES_COLUMNS = [
    "time_series_id",
    "monitoring_location_id",
    "parameter_code",
    "parameter_name",
    "parameter_description",
    "unit_of_measure",
    "sublocation_identifier",
    "web_description",
    "primary",
    "computation_identifier",
    "begin",
    "end",
]
SECONDS_PER_DAY = 86400
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
QUOTA_ERROR_NAME = "QuotaExhausted"


class RequestPacer:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self) -> None:
        with self.lock:
            pause = self.last + self.interval - time.monotonic()
            if pause > 0:
                time.sleep(pause)
            self.last = time.monotonic()


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def site_numbers(sites_file: Path) -> list[str]:
    with open(sites_file, encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    numbers = [site["site_no"] for layer in document["layers"].values() for site in layer]
    if len(set(numbers)) != len(numbers):
        raise ValueError("a site appears in more than one layer")
    return numbers


def is_quota_error(error: Exception) -> bool:
    return type(error).__name__ == QUOTA_ERROR_NAME


def with_backoff(
    call,
    attempts: int,
    base_seconds: float,
    quota_wait_seconds: float = 0.0,
    quota_attempts: int = 0,
):
    quota_hits = 0
    attempt = 0
    while True:
        try:
            return call()
        except Exception as error:
            if is_quota_error(error) and quota_hits < quota_attempts:
                quota_hits += 1
                time.sleep(quota_wait_seconds)
                continue
            attempt += 1
            if attempt >= attempts:
                raise
            time.sleep(base_seconds * 2 ** (attempt - 1))


def retrying(call, config: dict):
    return with_backoff(
        call,
        config["retry_attempts"],
        config["retry_base_seconds"],
        config["quota_wait_seconds"],
        config["quota_attempts"],
    )


def fetch_catalog(numbers: list[str], config: dict, pacer: RequestPacer) -> pd.DataFrame:
    ids = [f"{SITE_PREFIX}{number}" for number in numbers]

    def call() -> pd.DataFrame:
        pacer.wait()
        frame, _ = waterdata.get_time_series_metadata(monitoring_location_id=ids, limit=10000)
        return frame

    frame = retrying(call, config)
    return pd.DataFrame(frame.drop(columns="geometry", errors="ignore"))


def select_series(catalog: pd.DataFrame, config: dict) -> pd.DataFrame:
    instantaneous = catalog[catalog["computation_identifier"] == config["catalog_computation"]]
    fixed = instantaneous["parameter_code"].isin(list(config["fixed_parameters"]))
    names = instantaneous["parameter_name"].fillna("")
    fluorescence = names.str.contains(config["fluorescence_name_pattern"], regex=True)
    selected = instantaneous[fixed | fluorescence].copy()
    start = pd.Timestamp(config["start_date"], tz="UTC")
    selected = selected[pd.to_datetime(selected["end"], utc=True) >= start]
    selected["site_no"] = selected["monitoring_location_id"].str.removeprefix(SITE_PREFIX)
    ordered = selected.sort_values(["site_no", "parameter_code", "time_series_id"])
    return ordered.reset_index(drop=True)


def missing_fixed_series(selected: pd.DataFrame, numbers: list[str], config: dict) -> pd.DataFrame:
    present = set(zip(selected["site_no"], selected["parameter_code"], strict=True))
    rows = [
        {"site_no": number, "parameter_code": code, "role": role}
        for number in numbers
        for code, role in config["fixed_parameters"].items()
        if (number, code) not in present
    ]
    return pd.DataFrame(rows, columns=["site_no", "parameter_code", "role"])


def year_window(year: int, run_end: pd.Timestamp) -> str:
    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    year_end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59, tz="UTC")
    end = min(year_end, run_end)
    return f"{start.strftime(TIMESTAMP_FORMAT)}/{end.strftime(TIMESTAMP_FORMAT)}"


def plan_site_years(
    selected: pd.DataFrame, config: dict, run_end: pd.Timestamp
) -> list[tuple[str, int, tuple[str, ...]]]:
    plan = []
    first_year = pd.Timestamp(config["start_date"]).year
    for site, group in selected.groupby("site_no"):
        begin = pd.to_datetime(group["begin"], utc=True).min().year
        last = min(pd.to_datetime(group["end"], utc=True).max(), run_end).year
        codes = tuple(sorted(group["parameter_code"].unique()))
        for year in range(max(first_year, begin), last + 1):
            plan.append((str(site), year, codes))
    return plan


def estimate_rows(selected: pd.DataFrame, config: dict, run_end: pd.Timestamp) -> int:
    start = pd.Timestamp(config["start_date"], tz="UTC")
    begin = pd.to_datetime(selected["begin"], utc=True).clip(lower=start)
    end = pd.to_datetime(selected["end"], utc=True).clip(upper=run_end)
    seconds = (end - begin).dt.total_seconds().clip(lower=0)
    return int((seconds / config["nominal_interval_seconds"]).sum())


def fetch_site_year(
    site: str, year: int, codes: tuple[str, ...], run_end: pd.Timestamp, config: dict, pacer
) -> pd.DataFrame:
    def call() -> pd.DataFrame:
        pacer.wait()
        frame, _ = waterdata.get_continuous(
            monitoring_location_id=f"{SITE_PREFIX}{site}",
            parameter_code=list(codes),
            statistic_id=config["statistic_id"],
            properties=config["response_properties"],
            time=year_window(year, run_end),
            limit=config["page_limit"],
            convert_type=False,
        )
        return pd.DataFrame(frame.drop(columns="geometry", errors="ignore"))

    return retrying(call, config)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def process_site_year(
    job: tuple[str, int, tuple[str, ...]], run_dir: Path, run_end: pd.Timestamp, config: dict, pacer
) -> dict:
    site, year, codes = job
    name = f"{site}_{year}.parquet"
    response_path = run_dir / "responses" / site / name
    tidy_path = run_dir / "tidy" / site / name
    try:
        if not response_path.exists():
            write_parquet(fetch_site_year(site, year, codes, run_end, config, pacer), response_path)
        if not tidy_path.exists():
            response = pd.read_parquet(response_path)
            if response.empty:
                tidy = pd.DataFrame({column: pd.Series(dtype="object") for column in TIDY_COLUMNS})
            else:
                tidy = tidy_response(response, config["source_column"])
            write_parquet(tidy, tidy_path)
    except Exception as error:
        return {"site_no": site, "year": year, "error": f"{type(error).__name__}: {error}"}
    return {"site_no": site, "year": year, "error": None}


def build_manifest(
    run_dir: Path, state: dict, config: dict, numbers: list[str], failures: list[dict]
) -> dict:
    files = []
    for path in sorted((run_dir / "tidy").rglob("*.parquet")):
        files.append(
            {
                "path": str(path.relative_to(run_dir)),
                "sha256": file_sha256(path),
                "rows": pq.ParquetFile(path).metadata.num_rows,
                "bytes": path.stat().st_size,
            }
        )
    return {
        "run_id": state["run_id"],
        "run_started_utc": state["started_utc"],
        "run_manifest_written_utc": datetime.now(UTC).strftime(TIMESTAMP_FORMAT),
        "service_endpoint": config["endpoint"],
        "catalog_endpoint": config["catalog_endpoint"],
        "client": {
            "package": "dataretrieval",
            "version": dataretrieval.__version__,
            "function": "dataretrieval.waterdata.get_continuous",
            "catalog_function": "dataretrieval.waterdata.get_time_series_metadata",
        },
        "query": {
            "sites": numbers,
            "start_date": config["start_date"],
            "end_utc": state["started_utc"],
            "statistic_id": config["statistic_id"],
            "fixed_parameters": config["fixed_parameters"],
            "fluorescence_name_pattern": config["fluorescence_name_pattern"],
            "granularity": "one request per site and calendar year, all selected codes",
        },
        "failures": failures,
        "total_rows": sum(entry["rows"] for entry in files),
        "total_bytes": sum(entry["bytes"] for entry in files),
        "files": files,
    }


def load_or_create_state(run_dir: Path, run_id: str) -> dict:
    path = run_dir / "run_state.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    state = {"run_id": run_id, "started_utc": datetime.now(UTC).strftime(TIMESTAMP_FORMAT)}
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--run-id", default=f"iv_{datetime.now(UTC):%Y%m%d}")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--sites", nargs="+")
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    raw_root = args.raw_root or REPO_ROOT / config["raw_root"]
    run_dir = raw_root / args.run_id
    numbers = args.sites or site_numbers(REPO_ROOT / config["sites_file"])
    pacer = RequestPacer(config["min_request_interval_seconds"])
    state = load_or_create_state(run_dir, args.run_id)
    run_end = pd.Timestamp(state["started_utc"])
    try:
        catalog = fetch_catalog(numbers, config, pacer)
    except Exception as error:
        raise SystemExit(f"USGS service unreachable: {type(error).__name__}: {error}") from error
    selected = select_series(catalog, config)
    missing = missing_fixed_series(selected, numbers, config)
    plan = plan_site_years(selected, config, run_end)
    if args.years:
        plan = [job for job in plan if job[1] in args.years]
    print(f"series: {len(selected)}, site-years: {len(plan)}")
    print(
        f"estimated rows (nominal interval upper bound): {estimate_rows(selected, config, run_end)}"
    )
    if args.estimate_only:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    selected[["site_no", *SERIES_COLUMNS]].to_csv(run_dir / "series_catalog.csv", index=False)
    missing.to_csv(run_dir / "missing_series.csv", index=False)
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        results = list(
            pool.map(lambda job: process_site_year(job, run_dir, run_end, config, pacer), plan)
        )
    failures = [result for result in results if result["error"]]
    manifest = build_manifest(run_dir, state, config, numbers, failures)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"rows: {manifest['total_rows']}, bytes: {manifest['total_bytes']}")
    print(f"failures: {len(failures)}")
