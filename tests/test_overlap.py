import pandas as pd

from rivwq.analysis import overlap

RULES = {
    "min_overlap_years": 3.0,
    "overlap_train_first_year": 2018,
    "overlap_train_last_year": 2024,
    "overlap_test_year": 2025,
    "min_train_year_coverage": 0.5,
    "min_test_coverage": 0.8,
    "min_fchl_years": 3.0,
}


def frame(rows: dict[str, dict]) -> pd.DataFrame:
    base = {
        "legacy_do_waiver": False,
        "turbidity_estimated_years": "",
        "dissolved_oxygen_estimated_years": "",
    }
    for year in range(2018, 2026):
        base[f"turb_cov_{year}"] = 1.0
        base[f"do_cov_{year}"] = 1.0
    data = {site: {**base, **values} for site, values in rows.items()}
    return pd.DataFrame.from_dict(data, orient="index")


def test_parse_and_format_years_round_trip():
    assert overlap.parse_years("2024;2025") == {2024, 2025}
    assert overlap.parse_years("") == set()
    assert overlap.parse_years(float("nan")) == set()
    assert overlap.format_years([2025, 2024]) == "2024;2025"


def test_train_threshold_is_inclusive_and_needs_three_years():
    values = {f"turb_cov_{year}": 0.0 for year in range(2018, 2025)}
    values.update({"turb_cov_2018": 0.5, "turb_cov_2019": 0.5, "turb_cov_2020": 0.49})
    table = frame({"two": values, "three": {**values, "turb_cov_2020": 0.5}})
    result = overlap.add_overlap_columns(table, RULES)
    assert list(result["train_years"]) == [2, 3]
    assert list(result["pass_overlap"]) == [False, True]


def test_test_year_threshold_is_inclusive():
    table = frame({"low": {"turb_cov_2025": 0.79}, "edge": {"turb_cov_2025": 0.8}})
    result = overlap.add_overlap_columns(table, RULES)
    assert list(result["test_ok"]) == [False, True]
    assert list(result["pass_overlap"]) == [False, True]


def test_dissolved_oxygen_required_unless_waived():
    values = {f"do_cov_{year}": 0.0 for year in range(2018, 2026)}
    table = frame({"needs": values, "waived": {**values, "legacy_do_waiver": True}})
    result = overlap.add_overlap_columns(table, RULES)
    assert list(result["pass_overlap"]) == [False, True]


def test_missing_values_count_as_not_covered():
    table = frame({"gap": {"turb_cov_2025": float("nan")}})
    assert not overlap.add_overlap_columns(table, RULES)["pass_overlap"].iloc[0]


def test_conservative_rule_drops_estimated_years():
    estimated = {"turbidity_estimated_years": "2018;2019;2025"}
    table = frame({"sampled": estimated, "measured": {}})
    result = overlap.add_overlap_columns(table, RULES)
    assert list(result["pass_overlap"]) == [True, True]
    assert list(result["train_years"]) == [7, 7]
    assert list(result["train_years_conservative"]) == [5, 7]
    assert list(result["pass_overlap_conservative"]) == [False, True]
    assert list(result["coverage_estimated"]) == [True, False]
    assert list(overlap.changed_by_conservative(result).index) == ["sampled"]


def test_estimated_dissolved_oxygen_only_counts_when_required():
    estimated = {"dissolved_oxygen_estimated_years": "2025"}
    table = frame({"needs": estimated, "skipped": {**estimated, "legacy_do_waiver": True}})
    result = overlap.add_overlap_columns(table, RULES)
    assert list(result["coverage_estimated"]) == [True, False]


def test_hydraulics_rule_needs_discharge_and_gage_coverage():
    values = {}
    for year in range(2018, 2026):
        values[f"discharge_cov_{year}"] = 1.0
        values[f"gage_cov_{year}"] = 1.0
    table = frame({"full": values, "no_gage": {**values, "gage_cov_2025": 0.2}, "absent": {}})
    result = overlap.add_overlap_columns(table, RULES)
    assert list(result["pass_overlap_hydraulics"]) == [True, False, False]


def target_frame(rows: dict[str, dict]) -> pd.DataFrame:
    base = {"fchl_before_2025": True, "fchl_years": 5.0}
    for year in range(2018, 2026):
        base[f"fchl_cov_{year}"] = 1.0
    data = {site: {**base, **values} for site, values in rows.items()}
    return overlap.add_target_columns(frame(data), RULES)


def test_target_flags_use_each_series_on_its_own():
    no_do = {f"do_cov_{year}": 0.0 for year in range(2018, 2026)}
    no_fchl = {f"fchl_cov_{year}": 0.0 for year in range(2018, 2026)}
    result = target_frame({"all": {}, "no_do": no_do, "no_fchl": no_fchl})
    assert list(result["target_turbidity_ok"]) == [True, True, True]
    assert list(result["target_do_ok"]) == [True, False, True]
    assert list(result["target_fchl_ok"]) == [True, True, False]
    assert list(result["train_years_do"]) == [7, 0, 7]


def test_fchl_target_needs_years_before_2025_and_enough_years():
    result = target_frame(
        {
            "late": {"fchl_before_2025": False},
            "short": {"fchl_years": 2.9},
            "edge": {"fchl_years": 3.0},
        }
    )
    assert list(result["target_fchl_ok"]) == [False, False, True]


def test_target_coverage_thresholds_are_inclusive():
    train = {f"do_cov_{year}": 0.5 for year in (2018, 2019, 2020)}
    train.update({f"do_cov_{year}": 0.0 for year in range(2021, 2025)})
    result = target_frame(
        {
            "edge": {**train, "do_cov_2025": 0.8},
            "low_test": {**train, "do_cov_2025": 0.79},
            "two_years": {**train, "do_cov_2020": 0.49, "do_cov_2025": 0.8},
        }
    )
    assert list(result["target_do_ok"]) == [True, False, False]


def test_waiver_does_not_change_per_target_flags():
    no_do = {f"do_cov_{year}": 0.0 for year in range(2018, 2026)}
    waived = target_frame({"waived": {**no_do, "legacy_do_waiver": True}})
    assert not waived["target_do_ok"].iloc[0]


def test_turbidity_only_eligible_and_summary():
    no_do = {f"do_cov_{year}": 0.0 for year in range(2018, 2026)}
    no_fchl = {f"fchl_cov_{year}": 0.0 for year in range(2018, 2026)}
    result = target_frame({"all": {}, "only": {**no_do, **no_fchl}, "do": no_fchl})
    assert list(overlap.turbidity_only_eligible(result).index) == ["only"]
    summary = overlap.target_summary(result).set_index("group")["sites"]
    assert summary["target_turbidity_ok"] == 3
    assert summary["target_do_ok"] == 2
    assert summary["target_fchl_ok"] == 1
    assert summary["all three targets"] == 1
    assert summary["turbidity only"] == 1
