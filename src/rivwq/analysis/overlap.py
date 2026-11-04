from __future__ import annotations

import pandas as pd

SERIES = {
    "turbidity": "turb_cov",
    "dissolved_oxygen": "do_cov",
    "discharge": "discharge_cov",
    "gage": "gage_cov",
    "fchl": "fchl_cov",
}
TARGETS = {"turbidity": "turbidity", "dissolved_oxygen": "do", "fchl": "fchl"}


def parse_years(text: object) -> set[int]:
    if not isinstance(text, str) or not text:
        return set()
    return {int(part) for part in text.split(";")}


def format_years(years: list[int]) -> str:
    return ";".join(str(year) for year in sorted(years))


def estimated_mask(table: pd.DataFrame, variable: str, year: int) -> pd.Series:
    column = f"{variable}_estimated_years"
    if column not in table.columns:
        return pd.Series(False, index=table.index)
    return table[column].apply(lambda text: year in parse_years(text)).astype(bool)


def year_ok(
    table: pd.DataFrame, variable: str, year: int, minimum: float, exclude_estimated: bool
) -> pd.Series:
    column = f"{SERIES[variable]}_{year}"
    if column not in table.columns:
        return pd.Series(False, index=table.index)
    result = (table[column] >= minimum).fillna(False).astype(bool)
    if exclude_estimated:
        result &= ~estimated_mask(table, variable, year)
    return result


def required_year_ok(
    table: pd.DataFrame, year: int, minimum: float, exclude_estimated: bool
) -> pd.Series:
    turbidity = year_ok(table, "turbidity", year, minimum, exclude_estimated)
    oxygen = year_ok(table, "dissolved_oxygen", year, minimum, exclude_estimated)
    return turbidity & (oxygen | table["legacy_do_waiver"].astype(bool))


def train_years(
    table: pd.DataFrame, rules: dict, variables: list[str], exclude_estimated: bool
) -> pd.Series:
    years = range(rules["overlap_train_first_year"], rules["overlap_train_last_year"] + 1)
    count = pd.Series(0, index=table.index)
    for year in years:
        ok = pd.Series(True, index=table.index)
        for variable in variables:
            minimum = rules["min_train_year_coverage"]
            ok &= year_ok(table, variable, year, minimum, exclude_estimated)
        count += ok.astype(int)
    return count


def holdout_ok(
    table: pd.DataFrame, rules: dict, variables: list[str], exclude_estimated: bool
) -> pd.Series:
    ok = pd.Series(True, index=table.index)
    for variable in variables:
        minimum = rules["min_test_coverage"]
        ok &= year_ok(table, variable, rules["overlap_test_year"], minimum, exclude_estimated)
    return ok


def add_overlap_columns(table: pd.DataFrame, rules: dict) -> pd.DataFrame:
    result = table.copy()
    first, last = rules["overlap_train_first_year"], rules["overlap_train_last_year"]
    needs_oxygen = ~result["legacy_do_waiver"].astype(bool)
    for suffix, exclude in (("", False), ("_conservative", True)):
        count = pd.Series(0, index=result.index)
        for year in range(first, last + 1):
            ok = required_year_ok(result, year, rules["min_train_year_coverage"], exclude)
            count += ok.astype(int)
        passed_test = required_year_ok(
            result, rules["overlap_test_year"], rules["min_test_coverage"], exclude
        )
        result[f"train_years{suffix}"] = count
        result[f"test_ok{suffix}"] = passed_test
        result[f"pass_overlap{suffix}"] = (count >= rules["min_overlap_years"]) & passed_test
    result["coverage_estimated"] = estimated_in_rule_series(result, needs_oxygen, first, rules)
    hydraulic = ["discharge", "gage"]
    hydraulic_train = train_years(result, rules, hydraulic, False)
    hydraulic_test = holdout_ok(result, rules, hydraulic, False)
    result["pass_overlap_hydraulics"] = (
        result["pass_overlap"] & (hydraulic_train >= rules["min_overlap_years"]) & hydraulic_test
    )
    return result


def estimated_in_rule_series(
    table: pd.DataFrame, needs_oxygen: pd.Series, first_year: int, rules: dict
) -> pd.Series:
    years = range(first_year, rules["overlap_test_year"] + 1)
    flagged = pd.Series(False, index=table.index)
    for year in years:
        flagged |= estimated_mask(table, "turbidity", year)
        flagged |= estimated_mask(table, "dissolved_oxygen", year) & needs_oxygen
    return flagged


def changed_by_conservative(table: pd.DataFrame) -> pd.DataFrame:
    changed = table["pass_overlap"] != table["pass_overlap_conservative"]
    return table[changed]


def series_eligibility(
    table: pd.DataFrame, rules: dict, variable: str
) -> tuple[pd.Series, pd.Series]:
    count = train_years(table, rules, [variable], False)
    holdout = holdout_ok(table, rules, [variable], False)
    eligible = (count >= rules["min_overlap_years"]) & holdout
    return count, eligible


def add_target_columns(table: pd.DataFrame, rules: dict) -> pd.DataFrame:
    result = table.copy()
    for variable, name in TARGETS.items():
        count, eligible = series_eligibility(result, rules, variable)
        result[f"train_years_{name}"] = count
        result[f"target_{name}_ok"] = eligible
    fchl_rule = result["fchl_before_2025"].astype(bool) & (
        result["fchl_years"] >= rules["min_fchl_years"]
    )
    result["target_fchl_ok"] = result["target_fchl_ok"] & fchl_rule
    return result


def target_summary(table: pd.DataFrame) -> pd.DataFrame:
    flags = ["target_turbidity_ok", "target_do_ok", "target_fchl_ok"]
    rows = [{"group": "sites", "sites": len(table)}]
    rows += [{"group": name, "sites": int(table[name].sum())} for name in flags]
    rows.append({"group": "all three targets", "sites": int(table[flags].all(axis=1).sum())})
    rows.append({"group": "turbidity only", "sites": len(turbidity_only_eligible(table))})
    return pd.DataFrame(rows)


def turbidity_only_eligible(table: pd.DataFrame) -> pd.DataFrame:
    others = table[["target_do_ok", "target_fchl_ok"]].any(axis=1)
    return table[table["target_turbidity_ok"] & ~others]
