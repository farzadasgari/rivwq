import numpy as np
import pandas as pd
import pytest

from rivwq.usgs import qc_rules

CONFIG = qc_rules.load_config()


def build(values, code="63680", start="2025-01-01", freq="15min", qualifiers="", sensor="s1"):
    count = len(values)
    return pd.DataFrame(
        {
            "site_no": "0001",
            "datetime_utc": pd.date_range(start, periods=count, freq=freq, tz="UTC"),
            "parameter_code": code,
            "sensor_id": sensor,
            "value": np.asarray(values, dtype="float64"),
            "qualifiers": qualifiers,
            "approval_status": "Provisional",
            "unit_of_measure": "u",
            "source_column": "src",
        }
    )


def rule(frame, name, level="strict"):
    return qc_rules.apply_level(frame, CONFIG, level)[name].to_numpy()


def test_null_value_is_flagged():
    flags = rule(build([1.0, np.nan, 2.0]), "value_null")
    assert flags.tolist() == [False, True, False]


def test_unreliable_qualifier_matches_whole_tokens():
    frame = build([1.0, 1.0, 1.0, 1.0, 1.0])
    frame["qualifiers"] = ["", "EQUIP", "ESTIMATED|ICE", "ESTIMATED", "EQUIPMENT"]
    assert rule(frame, "unreliable_qualifier").tolist() == [False, True, True, False, False]


@pytest.mark.parametrize(
    ("code", "values", "expected"),
    [
        ("63680", [-0.1, 0.0, 5.0], [True, False, False]),
        ("00300", [-1.0, 0.0, 9.0], [True, False, False]),
        ("00400", [-0.5, 0.0, 14.0, 14.5], [True, False, False, True]),
        ("32316", [-1.0, 0.0, 3.0], [True, False, False]),
    ],
)
def test_physical_limits(code, values, expected):
    assert rule(build(values, code=code), "physical_limit").tolist() == expected


def test_negative_discharge_and_stage_are_reported_not_excluded():
    for code in ("00060", "00065"):
        flags = qc_rules.apply_level(build([-5.0, 3.0], code=code), CONFIG, "strict")
        assert flags["negative_discharge_stage"].tolist() == [True, False]
        assert flags["exclude"].tolist() == [False, False]


def test_negative_temperature_is_not_flagged():
    flags = qc_rules.apply_level(build([-1.0, 2.0], code="00010"), CONFIG, "strict")
    assert not flags["exclude"].any()


def test_temperature_zero_runs_need_minimum_length():
    minimum = CONFIG["temperature_zero_run"]["min_run_values"]
    values = [0.0] * (minimum - 1) + [1.0] + [0.0] * minimum + [2.0]
    flags = rule(build(values, code="00010"), "temperature_zero_run")
    assert flags[: minimum - 1].sum() == 0
    assert flags[minimum : 2 * minimum].all()
    assert not flags[-1]


def test_zero_run_is_broken_by_a_gap():
    minimum = CONFIG["temperature_zero_run"]["min_run_values"]
    first = build([0.0] * minimum, code="00010")
    second = build([0.0] * minimum, code="00010", start="2025-02-01")
    frame = pd.concat([first, second], ignore_index=True)
    assert rule(frame, "temperature_zero_run").all()
    short = pd.concat([first.iloc[: minimum - 1], second.iloc[: minimum - 1]], ignore_index=True)
    assert not rule(short, "temperature_zero_run").any()


def test_zero_rule_ignores_other_parameters():
    values = [0.0] * 10
    assert not rule(build(values, code="63680"), "temperature_zero_run").any()


def test_ceiling_is_off_when_no_ceiling_is_configured():
    frame = build([10.0, 4000.0, 4000.0])
    assert not rule(frame, "turbidity_ceiling").any()


def test_ceiling_applies_to_its_own_series_only():
    config = dict(CONFIG)
    config["turbidity_ceiling"] = {"parameter_code": "63680", "ceilings": {"s1": 100.0}}
    first = build([50.0, 100.0, 120.0], sensor="s1")
    second = build([50.0, 100.0, 120.0], sensor="s2")
    flags = qc_rules.apply_level(pd.concat([first, second], ignore_index=True), config, "strict")
    assert flags["turbidity_ceiling"].tolist() == [False, True, True, False, False, False]


def values_per_hour(freq):
    return 3600 // pd.Timedelta(freq).seconds


def test_flatline_limit_is_in_hours():
    hours = CONFIG["flatline"]["max_run_hours"]["00300"]
    limit = int(hours * values_per_hour("15min"))
    varied = list(np.linspace(5.0, 9.0, 20))
    long_run = varied + [7.0] * (limit + 1) + varied
    short_run = varied + [7.0] * limit + varied
    assert rule(build(long_run, code="00300"), "flatline").sum() == limit + 1
    assert rule(build(short_run, code="00300"), "flatline").sum() == 0


def test_flatline_limit_does_not_depend_on_the_sampling_interval():
    hours = CONFIG["flatline"]["max_run_hours"]["00300"]
    varied = list(np.linspace(5.0, 9.0, 20))
    for freq in ("5min", "15min", "60min"):
        per_hour = values_per_hour(freq)
        over = varied + [7.0] * (int(hours * per_hour) + 1) + varied
        under = varied + [7.0] * (int(hours * per_hour) - 1) + varied
        assert rule(build(over, code="00300", freq=freq), "flatline").any()
        assert not rule(build(under, code="00300", freq=freq), "flatline").any()


def test_flatline_limit_differs_by_parameter():
    limits = CONFIG["flatline"]["max_run_hours"]
    assert limits["00400"] > limits["00300"]
    count = int((limits["00300"] + 1) * values_per_hour("15min"))
    assert rule(build([7.0] * count, code="00300"), "flatline").all()
    assert not rule(build([7.0] * count, code="00400"), "flatline").any()


def test_flatline_does_not_apply_to_other_parameters():
    assert not rule(build([3.0] * 500, code="63680"), "flatline").any()


def test_levels_are_nested():
    frames = [
        build([0.0] * 6 + [-1.0, np.nan, 3.0], code="00010"),
        build([7.0] * 400 + [15.0], code="00400", qualifiers="MAINT"),
        build([7.0] * 200 + [-2.0], code="00300"),
    ]
    tidy = pd.concat(frames, ignore_index=True)
    excluded = {
        level: qc_rules.apply_level(tidy, CONFIG, level)["exclude"] for level in CONFIG["levels"]
    }
    assert (excluded["lenient"] <= excluded["standard"]).all()
    assert (excluded["standard"] <= excluded["strict"]).all()
    assert excluded["lenient"].sum() < excluded["standard"].sum() < excluded["strict"].sum()


def test_config_levels_are_nested_rule_sets():
    levels = CONFIG["levels"]
    assert set(levels["lenient"]) < set(levels["standard"]) < set(levels["strict"])
    assert set(levels["strict"]) <= set(qc_rules.RULES)


def test_lenient_has_no_run_rules():
    tidy = build([0.0] * 10, code="00010")
    flags = qc_rules.apply_level(tidy, CONFIG, "lenient")
    assert not flags["temperature_zero_run"].any()
    assert not flags["exclude"].any()


def test_input_is_unchanged_and_value_is_carried():
    tidy = pd.concat(
        [build([0.0] * 8, code="00010"), build([7.0, np.nan, -1.0], code="00300")],
        ignore_index=True,
    )
    before = tidy.copy(deep=True)
    flags = qc_rules.apply_level(tidy, CONFIG, "strict")
    pd.testing.assert_frame_equal(tidy, before)
    pd.testing.assert_series_equal(flags["value"], tidy["value"])
    assert (flags["approval_status"] == "Provisional").all()


def test_rules_do_not_depend_on_the_date():
    values = [0.0] * 6 + [1.0] + [7.0] * 300
    first = pd.concat(
        [build(values[:7], code="00010"), build(values[7:] + [-1.0], code="00300")],
        ignore_index=True,
    )
    shifted = first.copy()
    shifted["datetime_utc"] = shifted["datetime_utc"] + pd.DateOffset(years=5, days=37)
    for level in CONFIG["levels"]:
        original = qc_rules.apply_level(first, CONFIG, level)
        moved = qc_rules.apply_level(shifted, CONFIG, level)
        columns = qc_rules.RULES + qc_rules.REPORT_ONLY_RULES + ["exclude"]
        pd.testing.assert_frame_equal(original[columns], moved[columns])


def test_config_has_no_date_keys():
    rule_settings = {k: v for k, v in CONFIG.items() if k not in ("manifest_check", "output_dir")}
    text = repr(rule_settings).lower()
    for word in ("year", "date", "period", "start", "end_"):
        assert word not in text


def test_count_flags_per_rule_site_and_parameter():
    tidy = pd.concat(
        [build([-1.0, 2.0, np.nan], code="63680"), build([-4.0, 1.0], code="00060")],
        ignore_index=True,
    )
    flags = qc_rules.apply_level(tidy, CONFIG, "lenient")
    counts = qc_rules.count_flags(flags, "lenient").set_index(["parameter_code", "rule"])
    assert counts.loc[("63680", "physical_limit"), "flagged"] == 1
    assert counts.loc[("63680", "value_null"), "flagged"] == 1
    assert counts.loc[("63680", "exclude"), "flagged"] == 2
    assert counts.loc[("63680", "exclude"), "rows"] == 3
    assert counts.loc[("00060", "negative_discharge_stage"), "flagged"] == 1
    assert counts.loc[("00060", "exclude"), "flagged"] == 0
