from __future__ import annotations

import numpy as np
import pandas as pd

from rivwq.gee import checks, export

SETTINGS = export.load_config()["checks"] | {"sites": ["1"], "minimum_scenes_per_group": 2}


def make_rows(site: str, times: list[str], craft: list[str], values: list[float], valid: list[int]):
    rows = []
    for index, (stamp, ship, value, count) in enumerate(
        zip(times, craft, values, valid, strict=True)
    ):
        for variant in ("cs60", "cs50"):
            rows.append(
                {
                    "system_index": f"{site}_{index}",
                    "site_no": site,
                    "time_utc": pd.Timestamp(stamp, tz="UTC"),
                    "spacecraft": ship,
                    "roi": "reach",
                    "erosion_px": 1,
                    "cloud_variant": variant,
                    "valid_count": count,
                    "total_count": 100,
                    "valid_fraction": count / 100,
                    "B8A_median": value,
                    "B11_median": value,
                    "B12_median": value,
                }
            )
    return rows


def frame() -> pd.DataFrame:
    rows = make_rows(
        "1",
        ["2024-05-01", "2024-06-01", "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01"],
        ["S2A", "S2B", "S2A", "S2C", "S2C", "S2B"],
        [100.0, 100.0, 120.0, 130.0, 130.0, 100.0],
        [50, 0, 80, 70, 60, 50],
    )
    return pd.DataFrame(rows)


def test_scene_counts_split_by_spacecraft():
    table = checks.scene_counts(frame())
    row = table[(table["site_no"] == "1") & (table["year"] == 2025)].iloc[0]
    assert row["S2C"] == 2
    assert row["S2B"] == 1
    assert row["total"] == 4


def test_valid_pixel_loss_is_one_minus_valid_share():
    loss = checks.valid_pixel_loss(frame())
    row = loss[loss["cloud_variant"] == "cs60"].iloc[0]
    assert np.isclose(row["pixel_loss"], 1.0 - 310 / 600)


def test_zero_valid_site_year_listed_even_without_scenes():
    zero = checks.zero_valid_site_years(frame(), ["1", "2"], [2024, 2025], SETTINGS)
    assert set(zero["site_no"]) == {"2"}
    assert len(zero) == 2


def test_site_year_with_only_empty_scenes_is_zero_valid():
    rows = pd.DataFrame(
        make_rows("3", ["2024-05-01"], ["S2A"], [1.0], [0]),
    )
    zero = checks.zero_valid_site_years(rows, ["3"], [2024], SETTINGS)
    assert zero.iloc[0]["scenes"] == 1
    assert zero.iloc[0]["valid_scenes"] == 0


def test_sensor_jump_flags_large_s2c_difference_only():
    table = checks.sensor_jump(frame(), SETTINGS, ["B8A"])
    row = table[table["comparison"] == "S2C_vs_S2A_S2B_after_split"].iloc[0]
    assert np.isclose(row["relative_difference"], 130.0 / 110.0 - 1.0)
    assert row["flag"]


def test_sensor_jump_needs_enough_scenes():
    strict = SETTINGS | {"minimum_scenes_per_group": 50}
    table = checks.sensor_jump(frame(), strict, ["B8A"])
    assert not table["flag"].any()
    assert table["relative_difference"].isna().all()


def test_month_matched_difference_ignores_seasonal_offset():
    times = pd.to_datetime(["2025-01-05", "2025-01-15", "2025-07-05", "2025-07-15"], utc=True)
    left = pd.DataFrame({"time_utc": times, "B8A_median": [11.0, 11.0, 220.0, 220.0]})
    right = pd.DataFrame({"time_utc": times, "B8A_median": [10.0, 10.0, 200.0, 200.0]})
    difference, months = checks.month_matched_difference(left, right, "B8A_median", 2)
    assert months == 2
    assert np.isclose(difference, 0.1)


def test_month_matched_difference_skips_thin_months():
    times = pd.to_datetime(["2025-01-05", "2025-02-05"], utc=True)
    left = pd.DataFrame({"time_utc": times, "B8A_median": [1.0, 1.0]})
    difference, months = checks.month_matched_difference(left, left, "B8A_median", 2)
    assert months == 0
    assert np.isnan(difference)
