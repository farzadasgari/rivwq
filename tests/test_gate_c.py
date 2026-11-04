import pandas as pd
import pytest

from rivwq.analysis import gate_c

RULES = {
    "min_pixels_across": 10.0,
    "min_turbidity_coverage_2025": 0.8,
    "min_overlap_years": 3.0,
    "min_fchl_years": 3.0,
}
CONFIG = {
    "rules": {**RULES, "high_turbidity_share": 0.10, "top_sites": 2},
    "coverage": {"gap_months_limit": 6, "days_per_month": 30.4375},
}


def rule_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pixels_across": [10.0, 9.99, 20.0, 20.0],
            "turb_cov_2025": [0.8, 0.9, 0.79, 0.95],
            "fchl_before_2025": [True, True, False, True],
            "fchl_years": [3.0, 5.0, 5.0, 2.9],
        },
        index=pd.Index(["a", "b", "c", "d"], name="site"),
    )


def test_rule_thresholds_are_inclusive():
    result = gate_c.add_rule_columns(rule_table(), RULES)
    assert list(result["pass_width_10px"]) == [True, False, True, True]
    assert list(result["pass_turbidity_cov"]) == [True, True, False, True]
    assert list(result["pass_fchl_target"]) == [True, True, False, False]


def test_missing_values_fail_rules():
    table = rule_table()
    table["turb_cov_2025"] = float("nan")
    result = gate_c.add_rule_columns(table, RULES)
    assert not result["pass_turbidity_cov"].any()


def test_scene_estimates_multiply_cloud_share_and_coverage():
    table = pd.DataFrame({"dates_total": [1000], "cs_share": [0.4], "turb_cov_mean": [0.5]})
    result = gate_c.estimate_scenes(table)
    assert result["est_scenes_upper"][0] == pytest.approx(400.0)
    assert result["est_scenes_matched"][0] == pytest.approx(200.0)


def test_rule_counts_and_core_subset():
    table = gate_c.add_rule_columns(rule_table(), RULES)
    table["pass_overlap"] = [True, True, True, False]
    table["pass_overlap_conservative"] = [True, False, True, False]
    table["pass_overlap_hydraulics"] = False
    table["pass_overlap_catalog"] = True
    counts = gate_c.rule_counts(table).set_index("rule")["sites"]
    assert counts["pass_width_10px"] == 3
    assert counts["width + turbidity coverage + overlap"] == 1
    assert counts["width + turbidity coverage + overlap conservative"] == 1
    assert list(gate_c.passing_core(table).index) == ["a"]


def test_region_of_state_uses_config_lists():
    regions = {"West": ["CA"], "South": ["TX"]}
    assert gate_c.region_of_state("TX", regions) == "South"
    assert gate_c.region_of_state("XX", regions) == "unknown"


def test_gap_and_high_turbidity_selection_use_strict_limits():
    limit_days = 6 * 30.4375
    table = pd.DataFrame(
        {
            "turbidity_gap_days": [limit_days, limit_days + 1, limit_days + 5, 10.0],
            "turbidity_interior_gap_days": [limit_days, limit_days + 1, 3.0, 2.0],
            "turbidity_share_above_200": [0.10, 0.11, 0.0, 0.0],
        },
        index=["a", "b", "c", "d"],
    )
    assert list(gate_c.gap_sites(table, CONFIG).index) == ["b"]
    assert list(gate_c.edge_gap_sites(table, CONFIG).index) == ["c"]
    assert list(gate_c.high_turbidity_sites(table, CONFIG).index) == ["b"]


def test_best_wide_sites_only_ranks_sites_passing_width():
    table = pd.DataFrame(
        {
            "pass_width_10px": [True, False, True],
            "est_scenes_matched": [10.0, 99.0, 20.0],
        },
        index=["a", "b", "c"],
    )
    assert list(gate_c.best_wide_sites(table, 2).index) == ["c", "a"]


def test_stratum_counts_report_fchl_subset():
    table = pd.DataFrame(
        {
            "turbidity_class": ["turbid", "turbid", "clear"],
            "regulation_class": ["free_flowing"] * 3,
            "regime_class": ["rain_fed"] * 3,
            "cloud_class": ["sunnier"] * 3,
            "region": ["West"] * 3,
            "pass_fchl_target": [True, False, False],
        }
    )
    counts = gate_c.stratum_counts(table)
    turbid = counts[(counts["axis"] == "turbidity_class") & (counts["stratum"] == "turbid")]
    assert int(turbid["sites"].iloc[0]) == 2
    assert int(turbid["with_fchl"].iloc[0]) == 1


def test_rename_coverage_uses_short_labels():
    frame = pd.DataFrame(
        columns=["turbidity_cov_2025", "dissolved_oxygen_cov_mean", "fchl_cov_2018", "other"]
    )
    renamed = gate_c.rename_coverage(frame)
    assert list(renamed.columns) == ["turb_cov_2025", "do_cov_mean", "fchl_cov_2018", "other"]


def test_hydraulic_sites_need_width_and_turbidity_coverage():
    screened = pd.DataFrame({"pixels_across": [10.0, 9.0, 20.0, 20.0]}, index=list("abcd"))
    rows = pd.DataFrame(
        {"site_no": list("abcd"), "turbidity_cov_2025": [0.8, 0.9, 0.79, float("nan")]}
    )
    assert gate_c.hydraulic_sites(screened, rows, RULES) == ["a"]


def test_coverage_jobs_skip_missing_codes_and_deduplicate():
    code_map = {"a": {"turbidity": "63680", "fchl": None}, "b": {"turbidity": "63680"}}
    assert gate_c.coverage_jobs(code_map) == [("a", "63680"), ("b", "63680")]
