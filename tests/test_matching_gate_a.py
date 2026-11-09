import json

import numpy as np
import pandas as pd
import pytest

from rivwq.matching import gate_a, readings, table
from tests.test_matching_table import write_run

CONFIG = readings.load_config()
SETTINGS = CONFIG["gate_a"]


def dates(count, start="2025-01-01", step=2):
    return pd.Series(pd.date_range(start, periods=count, freq=f"{step}D", tz="UTC"))


def test_complete_run_is_accepted_and_partial_or_failed_runs_are_refused(tmp_path):
    complete = {"run_id": "r", "pending": [], "failed": {}}
    (tmp_path / "run_state.json").write_text(json.dumps(complete))
    gate_a.require_complete_run(tmp_path)
    (tmp_path / "run_state.json").write_text(json.dumps({**complete, "pending": ["a_2018"]}))
    with pytest.raises(gate_a.IncompleteRunError):
        gate_a.require_complete_run(tmp_path)
    (tmp_path / "run_state.json").write_text(json.dumps({**complete, "failed": {"a_2018": "x"}}))
    with pytest.raises(gate_a.IncompleteRunError):
        gate_a.require_complete_run(tmp_path)


def test_effective_n_of_independent_values_is_close_to_the_count():
    rng = np.random.default_rng(3)
    result = gate_a.effective_n(dates(300), pd.Series(rng.normal(size=300)), SETTINGS)
    assert result["dates"] == 300
    assert result["n_eff"] == pytest.approx(300, rel=0.25)


def test_effective_n_shrinks_for_autocorrelated_values():
    rng = np.random.default_rng(4)
    values = np.cumsum(rng.normal(size=300))
    result = gate_a.effective_n(dates(300), pd.Series(values), SETTINGS)
    assert result["autocorrelation"] > 0.9
    assert result["n_eff"] < 0.1 * 300


def test_negative_autocorrelation_is_clipped_to_zero():
    values = pd.Series([1.0, -1.0] * 50)
    result = gate_a.effective_n(dates(100), values, SETTINGS)
    assert result["autocorrelation"] < 0
    assert result["n_eff"] == 100


def test_pairs_further_apart_than_the_gap_limit_are_not_used():
    values = pd.Series(np.arange(50, dtype=float))
    result = gate_a.effective_n(dates(50, step=10), values, SETTINGS)
    assert result["pairs"] == 0
    assert np.isnan(result["n_eff"])


def test_effective_n_needs_three_dates():
    result = gate_a.effective_n(dates(2), pd.Series([1.0, 2.0]), SETTINGS)
    assert np.isnan(result["n_eff"])
    assert result["dates"] == 2


def test_same_day_scenes_count_once_in_dates():
    stamps = pd.Series(pd.to_datetime(["2025-01-01 10:00", "2025-01-01 10:01"], utc=True))
    result = gate_a.effective_n(gate_a.utc_dates(stamps), pd.Series([1.0, 3.0]), SETTINGS)
    assert result["dates"] == 1


def synthetic_table(scenes):
    rows = []
    for number, (stamp, valid, turbidity) in enumerate(scenes):
        for variant in ("nearest", "daily_mean"):
            rows.append(
                {
                    "site_no": "0001",
                    "system_index": f"s{number}",
                    "time_utc": pd.Timestamp(stamp, tz="UTC"),
                    "roi": "reach",
                    "erosion_px": 0,
                    "cloud_variant": "cs60",
                    "variant": variant,
                    "total_count": 100,
                    "valid_count": int(valid * 100),
                    "valid_fraction": valid,
                    "turbidity_n": 0 if np.isnan(turbidity) else 1,
                    "turbidity_value": turbidity,
                    "dissolved_oxygen_n": 1,
                    "dissolved_oxygen_value": 8.0,
                    "fchl_n": pd.NA,
                    "fchl_value": np.nan,
                    "qc_level": "standard",
                }
            )
    return pd.DataFrame(rows)


def test_funnel_counts_scenes_after_masks_and_target():
    scenes = [
        ("2025-01-01 10:00", 0.0, 5.0),
        ("2025-01-03 10:00", 0.3, np.nan),
        ("2025-01-05 10:00", 0.9, 7.0),
        ("2025-01-05 10:01", 0.9, 8.0),
    ]
    funnel = gate_a.funnel_rows(synthetic_table(scenes), CONFIG)
    nearest = funnel[funnel["variant"] == "nearest"].set_index("valid_fraction_min")
    assert nearest.loc[0.0, "scenes_total"] == 4
    assert nearest.loc[0.0, "scenes_after_masks"] == 3
    assert nearest.loc[0.5, "scenes_after_masks"] == 2
    assert nearest.loc[0.0, "turbidity_scenes"] == 2
    assert nearest.loc[0.0, "turbidity_dates"] == 1
    assert nearest.loc[0.0, "dissolved_oxygen_scenes"] == 3
    assert nearest.loc[0.0, "fchl_scenes"] == 0


def test_funnel_keeps_every_roi_erosion_cloud_and_variant_row():
    scenes = [("2025-01-01 10:00", 0.9, 5.0)]
    frame = synthetic_table(scenes)
    other = frame.assign(roi="extended", erosion_px=1, cloud_variant="scl")
    funnel = gate_a.funnel_rows(pd.concat([frame, other]), CONFIG)
    assert len(funnel) == 2 * 2 * len(SETTINGS["valid_fraction_levels"])


def test_decile_shares_show_concentration_in_matched_days():
    all_days = pd.Series(np.arange(1000, dtype=float))
    matched = pd.Series(np.arange(0, 100, dtype=float))
    shares = gate_a.decile_shares(all_days, matched, 10)
    assert shares["all_share"].tolist() == pytest.approx([0.1] * 10)
    assert shares["matched_share"].iloc[0] == 1.0
    assert shares["matched_share"].sum() == pytest.approx(1.0)


def test_selection_bias_is_small_for_unbiased_and_large_for_biased_days():
    days = pd.date_range("2025-01-01", periods=400, freq="D", tz="UTC")
    all_days = pd.Series(np.linspace(1, 100, 400), index=days)
    unbiased = [(str(day), 0.9, 1.0) for day in days[::4]]
    biased = [(str(day), 0.9, 1.0) for day in days[:100]]
    _, fair = gate_a.selection_bias(synthetic_table(unbiased), all_days, CONFIG)
    _, skewed = gate_a.selection_bias(synthetic_table(biased), all_days, CONFIG)
    fair_row = fair[fair["stage"] == "after_masks_with_turbidity"].iloc[0]
    skewed_row = skewed[skewed["stage"] == "after_masks_with_turbidity"].iloc[0]
    assert fair_row["ks_statistic"] < 0.05
    assert skewed_row["ks_statistic"] > 0.7
    assert skewed_row["median_matched"] < skewed_row["median_all"]


def test_selection_bias_reports_dates_without_turbidity_data():
    days = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")
    all_days = pd.Series(np.arange(5, dtype=float), index=days[:5])
    scenes = [(str(day), 0.9, 1.0) for day in days]
    _, summary = gate_a.selection_bias(synthetic_table(scenes), all_days, CONFIG)
    row = summary[summary["stage"] == "all_scene_dates"].iloc[0]
    assert row["dates"] == 10
    assert row["dates_without_turbidity"] == 5


def test_selection_stages_are_nested():
    scenes = [
        ("2025-01-01 10:00", 0.0, 5.0),
        ("2025-01-03 10:00", 0.9, np.nan),
        ("2025-01-05 10:00", 0.9, 7.0),
    ]
    stages = gate_a.selection_stages(synthetic_table(scenes), CONFIG)
    sizes = [len(stages[name]) for name in ("all_scene_dates", "after_masks")]
    assert sizes == [3, 2]
    assert len(stages["after_masks_with_turbidity"]) == 1


def test_report_runs_end_to_end_and_refuses_an_incomplete_run(tmp_path):
    repo, config = write_run(tmp_path)
    table.run(config, "run1", ["standard"], None, repo)
    state = repo / "gee" / "run1" / "run_state.json"
    state.write_text(json.dumps({"run_id": "run1", "pending": ["x"], "failed": {}}))
    with pytest.raises(gate_a.IncompleteRunError):
        gate_a.run(config, "run1", "standard", None, repo)
    state.write_text(json.dumps({"run_id": "run1", "pending": [], "failed": {}}))
    result = gate_a.run(config, "run1", "standard", None, repo)
    assert set(result) == {
        "gate_a_funnel",
        "gate_a_selection_deciles",
        "gate_a_selection_summary",
    }
    funnel = result["gate_a_funnel"]
    assert funnel["scenes_total"].max() == 2
    written = gate_a.write(result, tmp_path, "06_test_")
    assert all(path.exists() for path in written)
    assert not list((repo / "gee").rglob("06_*"))
