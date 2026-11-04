import pandas as pd

from rivwq.usgs import pool

REQUIRED = ["turbidity", "dissolved_oxygen", "water_temperature", "discharge", "gage_height"]
POOL_CONFIG = {"min_drainage_area_sqmi": 1000.0, "relaxed_missing_series": "dissolved_oxygen"}


def row(area: float, reasons: str, missing: tuple[str, ...] = ()) -> dict:
    values = {f"{name}_code": None if name in missing else "00001" for name in REQUIRED}
    return {"drainage_area_sqmi": area, "fail_reasons": reasons, **values}


def screened() -> pd.DataFrame:
    rows = {
        "01": row(2000.0, ""),
        "02": row(500.0, ""),
        "03": row(3000.0, "missing_required_series", ("dissolved_oxygen",)),
        "04": row(3000.0, "missing_required_series", ("dissolved_oxygen", "discharge")),
        "05": row(3000.0, "missing_required_series;overlap_too_short", ("dissolved_oxygen",)),
        "06": row(800.0, "missing_required_series", ("dissolved_oxygen",)),
        "07": row(3000.0, "missing_required_series", ("water_temperature",)),
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("site_no")


def test_relaxed_sites_fail_only_for_missing_dissolved_oxygen():
    found = pool.relaxed_sites(screened(), "dissolved_oxygen", REQUIRED)
    assert list(found) == ["03", "06"]


def test_pool_combines_large_candidates_and_relaxed_sites():
    table = screened()
    candidates = table.loc[["01", "02"]]
    result = pool.build_pool(candidates, table, {"01"}, POOL_CONFIG, REQUIRED)
    assert list(result.index) == ["01", "03"]
    assert list(result["relaxed_pool"]) == [False, True]
    assert list(result["already_done"]) == [True, False]


def test_relaxed_flag_applies_to_candidate_that_lacks_dissolved_oxygen():
    table = screened()
    candidates = table.loc[["03"]]
    result = pool.build_pool(candidates, table, set(), POOL_CONFIG, REQUIRED)
    assert list(result["relaxed_pool"]) == [True]


def test_normalise_code_pads_and_handles_missing():
    assert pool.normalise_code(32315.0) == "32315"
    assert pool.normalise_code("300") == "00300"
    assert pool.normalise_code(float("nan")) is None
    assert pool.normalise_code(None) is None
