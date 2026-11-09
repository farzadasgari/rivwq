import copy

import numpy as np
import pandas as pd
import pytest

from rivwq.matching import audits, readings
from rivwq.usgs import qc_rules
from tests.support_matching import regular, tidy_series

CONFIG = readings.load_config()
QC = qc_rules.load_config()
SETTINGS = {
    **CONFIG["audit"],
    "changeover_window_days": 60,
    "bootstrap_resamples": 200,
    "min_days_per_side": 10,
    "placebo_first_year": 2018,
    "placebo_last_year": 2020,
}


def test_native_summary_reports_30_and_60_minute_series():
    thirty = regular("2025-01-01", 100, "30min", site="a")
    sixty = regular("2025-01-01", 100, "60min", site="b")
    natives = readings.native_interval_table(pd.concat([thirty, sixty], ignore_index=True))
    summary = audits.native_summary(natives).set_index("site_no")
    assert summary.loc["a", "native_minutes_modal"] == 30
    assert summary.loc["b", "native_minutes_modal"] == 60


def test_native_summary_lists_every_interval_seen():
    early = regular("2024-06-01", 200, "15min")
    late = regular("2025-06-01", 100, "60min")
    natives = readings.native_interval_table(pd.concat([early, late], ignore_index=True))
    assert audits.native_summary(natives)["native_minutes_seen"].iloc[0] == "15|60"


def qualifier_frame():
    frame = regular("2025-01-01", 6, "15min", code="00060")
    frame["qualifiers"] = ["", "BACKWATER", "ESTIMATED|BACKWATER", "EQUIP", "REVISED", ""]
    frame.loc[5, "approval_status"] = "Provisional"
    return frame


def token_counts(frame):
    flags = qc_rules.apply_level(frame, QC, "standard")
    return audits.token_rows(frame, flags, QC["unreliable_qualifiers"])


def test_tokens_are_split_and_empty_is_counted_as_none():
    counts = token_counts(qualifier_frame())
    totals = counts.groupby("token")["rows"].sum().to_dict()
    assert totals == {"(none)": 2, "BACKWATER": 2, "ESTIMATED": 1, "EQUIP": 1, "REVISED": 1}


def test_kept_rows_follow_the_standard_qc_rules():
    counts = token_counts(qualifier_frame())
    kept = counts.groupby("token")["rows_kept"].sum().to_dict()
    assert kept["EQUIP"] == 0
    assert kept["BACKWATER"] == 2
    assert kept["REVISED"] == 1
    flagged = counts.groupby("token")["in_unreliable_list"].any().to_dict()
    assert flagged["EQUIP"] and not flagged["BACKWATER"]


def test_named_qualifiers_report_zero_for_absent_tokens():
    counts = token_counts(qualifier_frame())
    named = audits.named_qualifier_table(counts, CONFIG["audit"]["named_qualifiers"])
    rows = named.set_index("label")
    assert rows.loc["Bkw", "rows"] == 2
    assert rows.loc["Bkw", "tokens_found"] == "BACKWATER"
    assert rows.loc["Fld", "rows"] == 0
    assert rows.loc["Pmp", "rows"] == 0
    assert rows.loc["REVISED", "rows_kept_at_standard"] == 1


def test_qualifier_summary_counts_site_years():
    first = qualifier_frame()
    second = qualifier_frame().assign(site_no="0002")
    counts = pd.concat([token_counts(first), token_counts(second)], ignore_index=True)
    summary = audits.qualifier_summary(counts)
    assert summary[summary["token"] == "BACKWATER"]["site_years"].iloc[0] == 2


def test_provisional_share_by_year_and_parameter():
    first = regular("2024-12-31 23:00", 4, "15min", code="63680", status="Approved")
    second = regular("2025-01-01 00:00", 8, "15min", code="63680", status="Provisional")
    frame = pd.concat([first, second], ignore_index=True)
    flags = qc_rules.apply_level(frame, QC, "standard")
    counts = audits.provisional_counts(frame, flags, "Provisional")
    pooled = audits.provisional_by_year(counts).set_index("year")
    assert pooled.loc[2025, "provisional_share"] == 1.0
    assert pooled.loc[2024, "provisional_share"] == 0.0
    assert audits.provisional_in_year(counts, 2025)["provisional_share"].iloc[0] == 1.0


def test_provisional_share_ignores_qc_excluded_rows():
    frame = regular("2025-01-01", 4, "15min", code="63680", status="Provisional")
    frame.loc[0, "qualifiers"] = "EQUIP"
    frame.loc[1, "approval_status"] = "Approved"
    flags = qc_rules.apply_level(frame, QC, "standard")
    counts = audits.provisional_counts(frame, flags, "Provisional")
    assert counts["rows"].iloc[0] == 3
    assert counts["provisional_rows"].iloc[0] == 2


def test_do_coverage_for_year_uses_fifteen_minute_slots():
    frame = regular("2025-01-01", 96 * 365 // 2, "15min", code="00300", values=np.full(17520, 7.0))
    frame["value"] = np.linspace(5, 9, len(frame))
    flags = qc_rules.apply_level(frame, QC, "standard")
    result = audits.do_coverage_for_year(flags=flags, tidy=frame, code="00300", year=2025, cap=1.0)
    assert result["coverage"].iloc[0] == pytest.approx(0.5, abs=0.01)


def test_finalise_keeps_waived_site_without_do_series():
    frame = regular("2025-01-01", 96, "15min", code="00300", site="0001")
    flags = qc_rules.apply_level(frame, QC, "standard")
    coverage = audits.do_coverage_for_year(flags, frame, "00300", 2025, 1.0)
    combined = {
        "natives": readings.native_interval_table(frame),
        "spans": audits.sensor_spans(frame),
        "tokens": token_counts(qualifier_frame()),
        "provisional": audits.provisional_counts(frame, flags, "Provisional"),
        "do_coverage": coverage,
    }
    sites = {
        "0001": {"layer": "core", "cv_group": "a"},
        "0002": {"layer": "core", "cv_group": "b"},
    }
    result = audits.finalise(combined, sites, ["0001", "0002"], CONFIG)
    rows = result["do_waived_coverage"].set_index("site_no")
    assert bool(rows.loc["0001", "has_do_series"]) is True
    assert bool(rows.loc["0002", "has_do_series"]) is False


def test_waived_sites_reads_ids_with_leading_zeros(tmp_path):
    path = tmp_path / "waiver.csv"
    path.write_text("site,legacy_do_waiver\n01184000,True\n01646500,False\n09999999,True\n")
    config = copy.deepcopy(CONFIG)
    config["audit"]["waiver_table"] = "waiver.csv"
    assert audits.waived_sites(config, tmp_path, ["01184000", "01646500"]) == ["01184000"]


def two_sensor_turbidity(step, overlap=False):
    rng = np.random.default_rng(1)
    days = pd.date_range("2018-01-01", "2020-12-31", freq="D", tz="UTC")
    discharge = np.exp(5 + np.sin(np.arange(len(days)) / 20.0) + rng.normal(0, 0.2, len(days)))
    base = 2.0 * discharge**0.5 * np.exp(rng.normal(0, 0.05, len(days)))
    old_end = pd.Timestamp("2019-05-27", tz="UTC")
    new_start = pd.Timestamp("2019-07-24", tz="UTC") if not overlap else old_end
    new = days >= new_start
    turbidity = np.where(new, base * step, base)
    sensor = np.where(new, "new", "old")
    keep = (days <= old_end) | (days >= new_start)
    turbidity_frame = tidy_series(days[keep], values=turbidity[keep], code="63680")
    turbidity_frame["sensor_id"] = sensor[keep]
    discharge_frame = tidy_series(days, values=discharge, code="00060", sensor="q")
    return turbidity_frame, discharge_frame


def test_changeover_step_is_recovered_and_called_supported():
    turbidity, discharge = two_sensor_turbidity(step=1.5)
    result = audits.changeover_test(turbidity, discharge, SETTINGS)
    real = result[result["label"] == "changeover"].iloc[0]
    assert real["ratio"] == pytest.approx(1.5, rel=0.05)
    assert real["ci_low"] > 0
    assert not real["sensors_overlap"]
    assert real["old_sensor"] == "old"
    assert audits.step_reading(result) == "step supported"


def test_no_step_is_not_distinguishable_from_zero():
    turbidity, discharge = two_sensor_turbidity(step=1.0)
    result = audits.changeover_test(turbidity, discharge, SETTINGS)
    real = result[result["label"] == "changeover"].iloc[0]
    assert abs(real["coefficient"]) < 0.05
    assert audits.step_reading(result) == "not distinguishable from zero"


def test_placebo_years_have_no_changeover_and_near_zero_step():
    turbidity, discharge = two_sensor_turbidity(step=1.5)
    result = audits.changeover_test(turbidity, discharge, SETTINGS)
    placebo = result[result["label"].str.startswith("placebo")]
    assert set(placebo["label"]) == {"placebo_2018", "placebo_2020"}
    assert placebo["coefficient"].abs().max() < 0.1


def test_bootstrap_interval_is_reproducible():
    turbidity, discharge = two_sensor_turbidity(step=1.3)
    first = audits.changeover_test(turbidity, discharge, SETTINGS)
    second = audits.changeover_test(turbidity, discharge, SETTINGS)
    assert first.loc[0, "ci_low"] == second.loc[0, "ci_low"]
    assert first.loc[0, "ci_high"] == second.loc[0, "ci_high"]


def test_overlapping_sensors_are_reported_as_overlapping():
    turbidity, discharge = two_sensor_turbidity(step=1.0, overlap=True)
    late = turbidity[turbidity["sensor_id"] == "new"].iloc[[0]].copy()
    late["datetime_utc"] = pd.Timestamp("2019-05-26", tz="UTC")
    both = pd.concat([turbidity, late], ignore_index=True)
    result = audits.changeover_test(both, discharge, SETTINGS)
    assert bool(result.loc[0, "sensors_overlap"]) is True


def test_single_sensor_gives_no_changeover_table():
    turbidity, discharge = two_sensor_turbidity(step=1.0)
    turbidity["sensor_id"] = "only"
    assert audits.changeover_test(turbidity, discharge, SETTINGS).empty


def test_too_few_days_gives_no_coefficient():
    turbidity, discharge = two_sensor_turbidity(step=1.5)
    settings = {**SETTINGS, "min_days_per_side": 1000}
    result = audits.changeover_test(turbidity, discharge, settings)
    assert np.isnan(result.loc[0, "coefficient"])
    assert audits.step_reading(result) == "not testable: too few days"


def test_audit_site_runs_end_to_end_on_the_changeover_site():
    turbidity, discharge = two_sensor_turbidity(step=1.5)
    tidy = pd.concat([turbidity, discharge], ignore_index=True)
    config = copy.deepcopy(CONFIG)
    config["audit"] = {**SETTINGS, "changeover_site": "0001"}
    parts = audits.audit_site(tidy, config, QC)
    assert {"natives", "spans", "tokens", "provisional", "do_coverage", "changeover"} <= set(parts)
    assert parts["spans"]["sensor_id"].nunique() == 3
