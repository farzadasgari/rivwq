import pandas as pd
import pytest

from rivwq.usgs import flags

CONFIG = {
    "fchl_window": ["2018-01-01", "2026-01-01"],
    "fchl_out_of_time_start": "2025-01-01",
    "fchl_min_years_before_out_of_time": 1.0,
    "caveat_sites": {"07374000": "tidal"},
}


def table(rows: dict[str, tuple]) -> pd.DataFrame:
    frame = pd.DataFrame.from_dict(
        rows,
        orient="index",
        columns=[
            "chlorophyll_fluorescence_code",
            "chlorophyll_fluorescence_first",
            "chlorophyll_fluorescence_last",
            "dissolved_oxygen_code",
        ],
    )
    return frame.rename_axis("site_no")


def test_fchl_years_counted_inside_window():
    frame = table(
        {
            "a": ("32316", "2016-01-01", "2026-11-03", "300"),
            "b": ("32316", "2021-01-01", "2023-01-01", "300"),
        }
    )
    result = flags.add_fchl_columns(frame, CONFIG)
    assert result.at["a", "fchl_first"] == pd.Timestamp("2018-01-01")
    assert result.at["a", "fchl_last"] == pd.Timestamp("2026-01-01")
    assert result.at["a", "fchl_years"] == pytest.approx(8.0, abs=0.01)
    assert result.at["b", "fchl_years"] == pytest.approx(2.0, abs=0.01)


def test_fchl_started_in_2024_is_flagged_not_before_2025():
    frame = table(
        {
            "late": ("32316", "2024-12-04", "2026-11-03", "300"),
            "mid": ("32316", "2024-06-01", "2026-11-03", "300"),
            "early": ("32316", "2020-01-01", "2026-11-03", "300"),
        }
    )
    result = flags.add_fchl_columns(frame, CONFIG)
    assert not result.at["late", "fchl_before_2025"]
    assert not result.at["mid", "fchl_before_2025"]
    assert result.at["early", "fchl_before_2025"]


def test_one_full_year_before_2025_counts():
    frame = table({"edge": ("32316", "2024-01-01", "2026-11-03", "300")})
    assert flags.add_fchl_columns(frame, CONFIG).at["edge", "fchl_before_2025"]


def test_site_without_fchl_has_zero_years_and_false_flag():
    frame = table({"none": (None, None, None, "300")})
    result = flags.add_fchl_columns(frame, CONFIG)
    assert result.at["none", "fchl_years"] == 0.0
    assert pd.isna(result.at["none", "fchl_first"])
    assert not result.at["none", "fchl_before_2025"]


def test_fchl_entirely_after_window_has_zero_years():
    frame = table({"future": ("32316", "2026-02-01", "2026-11-03", "300")})
    assert flags.add_fchl_columns(frame, CONFIG).at["future", "fchl_years"] == 0.0


def test_turbidity_only_when_do_or_fchl_missing():
    frame = table(
        {
            "full": ("32316", "2020-01-01", "2026-01-01", "300"),
            "no_do": ("32316", "2020-01-01", "2026-01-01", None),
            "no_fchl": (None, None, None, "300"),
        }
    )
    result = flags.add_turbidity_only(frame)
    assert list(result["turbidity_only"]) == [False, True, True]


def test_caveats_marked_with_reason():
    frame = table(
        {
            "07374000": ("32316", "2020-01-01", "2026-01-01", "300"),
            "other": ("32316", "2020-01-01", "2026-01-01", "300"),
        }
    )
    result = flags.add_caveats(frame, CONFIG["caveat_sites"])
    assert list(result["caveat"]) == [True, False]
    assert list(result["caveat_reason"]) == ["tidal", ""]
