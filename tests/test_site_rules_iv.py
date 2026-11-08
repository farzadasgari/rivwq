import numpy as np
import pandas as pd
import pytest

from rivwq.analysis import gate_c, site_rules_iv
from rivwq.usgs import qc_rules

CONFIG = site_rules_iv.load_config()
RULES = gate_c.load_config()["rules"]
DAY = 96


def tidy_frame(
    slots, code="63680", sensor="s1", start="2025-03-01", freq="15min", qualifiers="", values=None
):
    times = pd.DatetimeIndex(slots) if not isinstance(slots, int) else None
    if times is None:
        times = pd.date_range(start, periods=slots, freq=freq, tz="UTC")
    count = len(times)
    return pd.DataFrame(
        {
            "site_no": "0001",
            "datetime_utc": times,
            "parameter_code": code,
            "sensor_id": sensor,
            "value": np.full(count, 5.0) if values is None else np.asarray(values, dtype=float),
            "qualifiers": qualifiers,
            "approval_status": "Approved",
            "unit_of_measure": "u",
            "source_column": "src",
        }
    )


def coverage_of(tidy, code="63680", year=2025):
    qc_config = qc_rules.load_config()
    flags = qc_rules.apply_level(tidy, qc_config, CONFIG["qc_level"])
    result = site_rules_iv.parameter_coverage(flags, tidy, code, [year], CONFIG["coverage"]["cap"])
    return result.iloc[0]


def test_expected_slots_use_the_interval_and_year_length():
    assert site_rules_iv.expected_slot_count(2025, 900.0) == 365 * DAY
    assert site_rules_iv.expected_slot_count(2024, 900.0) == 366 * DAY
    assert site_rules_iv.expected_slot_count(2025, 3600.0) == 365 * 24


def test_full_day_of_a_year_gives_one_over_365():
    row = coverage_of(tidy_frame(DAY))
    assert row["coverage"] == pytest.approx(1 / 365)
    assert row["slots_covered"] == DAY


def test_duplicate_timestamps_and_jitter_inside_a_slot_count_once():
    times = pd.to_datetime(
        ["2025-03-01 00:00", "2025-03-01 00:00", "2025-03-01 00:05", "2025-03-01 00:15"], utc=True
    )
    row = coverage_of(tidy_frame(list(times)))
    assert row["slots_covered"] == 2


def test_excluded_values_do_not_count_as_covered():
    tidy = tidy_frame(DAY)
    tidy.loc[:47, "qualifiers"] = "MAINT"
    tidy.loc[48:59, "value"] = np.nan
    row = coverage_of(tidy)
    assert row["slots_covered"] == DAY - 48 - 12


def test_negative_turbidity_is_excluded_by_the_standard_level():
    tidy = tidy_frame(DAY, values=np.where(np.arange(DAY) < 10, -1.0, 5.0))
    assert coverage_of(tidy)["slots_covered"] == DAY - 10


def test_values_outside_the_calendar_year_are_ignored():
    tidy = tidy_frame(4, start="2024-12-31 23:30")
    assert coverage_of(tidy, year=2025)["slots_covered"] == 2
    assert coverage_of(tidy, year=2024)["slots_covered"] == 2


def test_two_sensors_are_merged_by_time_without_double_counting_overlap():
    first = tidy_frame(48, sensor="old", start="2025-03-01 00:00")
    second = tidy_frame(48, sensor="new", start="2025-03-01 06:00")
    row = coverage_of(pd.concat([first, second], ignore_index=True))
    assert row["slots_covered"] == 48 + 48 - 24
    assert row["overlap_slots"] == 24
    assert row["sensors"] == 2


def test_changeover_gap_between_sensors_stays_uncovered():
    first = tidy_frame(24, sensor="old", start="2025-03-01 00:00")
    second = tidy_frame(24, sensor="new", start="2025-03-02 00:00")
    row = coverage_of(pd.concat([first, second], ignore_index=True))
    assert row["slots_covered"] == 48


def test_sensor_exclusion_in_one_sensor_is_filled_by_the_other():
    first = tidy_frame(24, sensor="old", qualifiers="EQUIP")
    second = tidy_frame(24, sensor="new")
    row = coverage_of(pd.concat([first, second], ignore_index=True))
    assert row["slots_covered"] == 24


def test_hourly_series_uses_its_own_interval():
    row = coverage_of(tidy_frame(24, freq="60min"))
    assert row["interval_minutes"] == 60.0
    assert row["coverage"] == pytest.approx(24 / (365 * 24))


def test_coverage_is_capped():
    tidy = tidy_frame(DAY)
    flags = qc_rules.apply_level(tidy, qc_rules.load_config(), "standard")
    result = site_rules_iv.parameter_coverage(flags, tidy, "63680", [2025], 0.001)
    assert result.iloc[0]["coverage"] == 0.001


def test_unsupported_coverage_definition_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("coverage:\n  year_basis: local\n  interval_statistic: median\n")
    with pytest.raises(site_rules_iv.SiteRulesError):
        site_rules_iv.load_config(path)


def old_table(sites):
    rows = {}
    for site in sites:
        row = {
            "station": site,
            "pixels_across": 20.0,
            "fchl_before_2025": False,
            "fchl_years": 0.0,
            "legacy_do_waiver": False,
            "caveat": False,
            "turbidity_estimated_years": "2025",
            "dissolved_oxygen_estimated_years": "",
            "coverage_estimated": True,
        }
        for prefix in ("turb_cov", "do_cov", "fchl_cov", "discharge_cov", "gage_cov"):
            for year in range(2018, 2026):
                row[f"{prefix}_{year}"] = 0.0
        rows[site] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("site_no")


def long_rows(site, target, coverage_by_year):
    return pd.DataFrame(
        [
            {"site_no": site, "target": target, "code": "x", "year": year, "coverage": value}
            for year, value in coverage_by_year.items()
        ]
    )


def full_years(value):
    return dict.fromkeys(range(2018, 2026), value)


def test_adapter_replaces_coverage_and_clears_estimated_flags():
    long = pd.concat(
        [
            long_rows("a", "turbidity", full_years(0.9)),
            long_rows("a", "dissolved_oxygen", full_years(0.9)),
        ]
    )
    wide = site_rules_iv.wide_coverage(long, ["a"], list(range(2018, 2026)))
    table = site_rules_iv.apply_rules(old_table(["a"]), wide, RULES)
    assert table.loc["a", "turb_cov_2025"] == 0.9
    assert bool(table.loc["a", "pass_turbidity_cov"])
    assert bool(table.loc["a", "pass_overlap"])
    assert bool(table.loc["a", "pass_overlap_conservative"])
    assert not bool(table.loc["a", "coverage_estimated"])


def test_adapter_uses_the_rule_thresholds_inclusively():
    long = pd.concat(
        [
            long_rows("edge", "turbidity", full_years(0.8)),
            long_rows("edge", "dissolved_oxygen", full_years(0.8)),
            long_rows("low", "turbidity", {**full_years(0.8), 2025: 0.7999}),
            long_rows("low", "dissolved_oxygen", full_years(0.8)),
        ]
    )
    wide = site_rules_iv.wide_coverage(long, ["edge", "low"], list(range(2018, 2026)))
    table = site_rules_iv.apply_rules(old_table(["edge", "low"]), wide, RULES)
    assert list(table["pass_turbidity_cov"]) == [True, False]
    assert list(table["pass_overlap"]) == [True, False]


def test_missing_series_counts_as_zero_coverage():
    long = long_rows("a", "turbidity", full_years(0.9))
    wide = site_rules_iv.wide_coverage(long, ["a"], list(range(2018, 2026)))
    table = site_rules_iv.apply_rules(old_table(["a"]), wide, RULES)
    assert not bool(table.loc["a", "pass_overlap"])
    assert table.loc["a", "do_cov_2025"] == 0.0


def test_compare_flags_marks_changed_sites():
    old = old_table(["a", "b"])
    new = old.copy()
    for table in (old, new):
        table["pass_width_10px"] = True
        table["pass_turbidity_cov"] = True
        table["pass_overlap"] = True
        table["pass_overlap_conservative"] = True
        table["target_turbidity_ok"] = True
        table["target_do_ok"] = True
        table["target_fchl_ok"] = False
        table["coverage_estimated"] = False
    new.loc["b", "pass_overlap"] = False
    comparison = site_rules_iv.compare_flags(old, new, {"a": "core", "b": "core"})
    assert list(comparison["changed"]) == [False, True]


def test_near_threshold_flags_values_within_the_margin_on_both_sides():
    long = pd.concat(
        [
            long_rows("a", "turbidity", {2025: 0.79, 2024: 0.51, 2023: 0.45, 2022: 0.9}),
            long_rows("a", "dissolved_oxygen", {2025: 0.82, 2024: 0.6}),
        ]
    )
    table = pd.DataFrame(
        {"legacy_do_waiver": [False], "target_fchl_ok": [False]}, index=pd.Index(["a"])
    )
    near = site_rules_iv.near_threshold_rows(long, table, RULES, 0.02)
    found = {(row.target, row.year): row.passes_year for row in near.itertuples()}
    assert found == {
        ("turbidity", 2025): False,
        ("turbidity", 2024): True,
        ("dissolved_oxygen", 2025): True,
    }


def test_near_threshold_keeps_dissolved_oxygen_for_waived_sites():
    long = long_rows("a", "dissolved_oxygen", {2025: 0.8})
    table = pd.DataFrame(
        {"legacy_do_waiver": [True], "target_fchl_ok": [False]}, index=pd.Index(["a"])
    )
    assert len(site_rules_iv.near_threshold_rows(long, table, RULES, 0.02)) == 1


def test_near_threshold_skips_fchl_when_not_a_target():
    long = long_rows("a", "fchl", {2025: 0.8})
    table = pd.DataFrame(
        {"legacy_do_waiver": [False], "target_fchl_ok": [False]}, index=pd.Index(["a"])
    )
    assert site_rules_iv.near_threshold_rows(long, table, RULES, 0.02).empty


def test_frozen_table_replaces_only_the_recomputed_rows_and_keeps_order():
    old = old_table(["a", "b", "c"])
    new = old.loc[["b"]].copy()
    new["pixels_across"] = 99.0
    result = site_rules_iv.frozen_table(old, new)
    assert list(result.index) == ["a", "b", "c"]
    assert list(result["pixels_across"]) == [20.0, 99.0, 20.0]
    assert list(result.columns) == list(old.columns)


def test_frozen_table_rejects_missing_columns():
    old = old_table(["a", "b"])
    new = old.loc[["b"]].drop(columns=["pixels_across"])
    with pytest.raises(site_rules_iv.SiteRulesError):
        site_rules_iv.frozen_table(old, new)


def test_scene_estimates_follow_the_new_mean_coverage():
    table = old_table(["a"])
    table["dates_total"] = 1000.0
    table["cs_share"] = 0.5
    for year in range(2018, 2026):
        table[f"turb_cov_{year}"] = 0.8 if year > 2021 else 0.4
    result = site_rules_iv.refresh_scene_estimates(table, list(range(2018, 2026)))
    assert result.loc["a", "turb_cov_mean"] == pytest.approx(0.6)
    assert result.loc["a", "est_scenes_upper"] == pytest.approx(500.0)
    assert result.loc["a", "est_scenes_matched"] == pytest.approx(300.0)
