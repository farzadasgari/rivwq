import os

import pandas as pd

from rivwq.usgs import manifest


def make_run(tmp_path):
    entries = []
    for name, stamp in [("0001_2019", 1_600_000_000), ("0001_2020", 1_700_000_000)]:
        response = tmp_path / "responses" / "0001" / f"{name}.parquet"
        response.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"a": [1]}).to_parquet(response)
        os.utime(response, (stamp, stamp))
        entries.append({"path": f"tidy/0001/{name}.parquet", "sha256": "x", "rows": 1})
    return {"files": entries, "total_rows": 2}


def test_fetched_at_comes_from_cache_mtime_and_marks_refetched(tmp_path):
    original = make_run(tmp_path)
    result = manifest.add_fetched_at(original, tmp_path, ["0001_2020"])
    assert result["files"][0]["fetched_at"] == "2020-09-13T12:26:40Z"
    assert result["files"][1]["fetched_at"] == "2023-11-14T22:13:20Z"
    assert [entry["refetched"] for entry in result["files"]] == [False, True]
    assert result["files"][0]["sha256"] == "x"
    assert result["total_rows"] == 2
    assert "fetched_at" not in original["files"][0]


def test_candidates_exclude_refetched_and_final_year():
    files = [{"path": f"tidy/s/s_{year}.parquet"} for year in (2018, 2019, 2020, 2026)]
    names = manifest.candidate_site_years({"files": files}, ["s_2019"], 2026)
    assert names == ["s_2018", "s_2020"]


def test_pick_is_reproducible_and_unique():
    names = [f"s_{year}" for year in range(2018, 2026)]
    first = manifest.pick_site_years(names, 7, 2)
    assert first == manifest.pick_site_years(list(reversed(names)), 7, 2)
    assert len(set(first)) == 2
