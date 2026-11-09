from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from rivwq.gee import export, stats

CONFIG = export.load_config()


def test_year_bounds_clip_to_start_and_run_time():
    run_start = datetime(2026, 11, 8, 10, 30, tzinfo=UTC)
    assert export.year_bounds(2018, "2018-01-01", run_start)[0] == "2018-01-01T00:00:00Z"
    assert export.year_bounds(2026, "2018-01-01", run_start)[1] == "2026-11-08T10:30:00Z"
    assert export.year_bounds(2023, "2018-01-01", run_start)[1] == "2024-01-01T00:00:00Z"


def test_quota_error_waits_and_then_succeeds():
    calls = {"count": 0}
    waits = []

    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("Quota exceeded: too many requests")
        return "ok"

    retry = {"attempts": 2, "backoff_seconds": 1, "quota_wait_seconds": 600, "quota_attempts": 5}
    assert export.with_retries(flaky, retry, sleep=waits.append) == "ok"
    assert waits == [600, 600]


def test_ordinary_error_is_raised_after_attempts():
    retry = {"attempts": 3, "backoff_seconds": 1, "quota_wait_seconds": 1, "quota_attempts": 2}
    waits = []

    def broken():
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        export.with_retries(broken, retry, sleep=waits.append)
    assert waits == [1, 2]


def test_empty_site_year_keeps_schema():
    assert list(export.tidy_site_year([], "123", CONFIG).columns) == stats.long_columns(CONFIG)


def test_atomic_parquet_leaves_no_partial_file(tmp_path):
    path = tmp_path / "tidy" / "1" / "1_2023.parquet"
    export.write_parquet(pd.DataFrame({"a": [1, 2]}), path)
    assert path.exists()
    assert not list(path.parent.glob("*.part"))


def test_raw_response_roundtrip(tmp_path):
    path = tmp_path / "responses" / "1" / "1_2023.json.gz"
    export.write_raw([{"properties": {"a": 1}}], path)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert json.load(handle) == [{"properties": {"a": 1}}]


def test_plan_jobs_one_per_site_and_year():
    assert export.plan_jobs(["a", "b"], [2018, 2019]) == [
        ("a", 2018),
        ("a", 2019),
        ("b", 2018),
        ("b", 2019),
    ]


@pytest.mark.gee
def test_live_assets_and_static_water():
    from rivwq.gee import session

    session.initialize()
    verified = export.verify_assets(CONFIG)
    assert verified["surface_water"]["type"] == "IMAGE"
    assert verified["surface_reflectance"]["type"] == "IMAGE_COLLECTION"
    assert verified["cloud_score"]["type"] == "IMAGE_COLLECTION"
