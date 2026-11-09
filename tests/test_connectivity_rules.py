from __future__ import annotations

import numpy as np
import pandas as pd

from rivwq.gee import connectivity_rules as rules

SETTINGS = rules.load_config()


def long_rows(variant: str, site: str, near: float, reach: float, extended: float) -> list[dict]:
    return [
        {"variant": variant, "site_no": site, "roi": "near_gauge", "area_km2": near},
        {"variant": variant, "site_no": site, "roi": "reach", "area_km2": reach},
        {"variant": variant, "site_no": site, "roi": "extended", "area_km2": extended},
    ]


def sample_long() -> pd.DataFrame:
    rows = []
    cases = (
        ("a", (1, 2, 2), (1, 2, 8), (1, 2, 8)),
        ("b", (1, 2, 2.2), (1, 2, 6), (1, 2, 6)),
        ("c", (1, 2, 14), (1, 2, 16), (1, 2, 8)),
        ("d", (0.05, 2, 8), (0.05, 2, 8), (0.05, 2, 8)),
    )
    for site, current, closed, capped in cases:
        rows += long_rows("current", site, *current)
        for k in SETTINGS["gap_join_pixels"]:
            rows += long_rows(rules.variant_name("A", k), site, *closed)
            rows += long_rows(rules.variant_name("B", k), site, *capped)
    return pd.DataFrame(rows)


def relaxed_settings(minimum: int) -> dict:
    return {**SETTINGS, "selection": {**SETTINGS["selection"], "min_fixed_sites": minimum}}


def test_flags_and_ratios():
    table = rules.wide_table(sample_long(), SETTINGS)
    current = table[table["variant"] == "current"].set_index("site_no")
    assert current.loc["a", "ratio_20_5"] == 1.0
    assert current.loc["a", "flag_broken"]
    assert current.loc["b", "flag_broken"]
    assert not current.loc["d", "flag_broken"]
    assert current.loc["c", "flag_leakage"]
    assert current.loc["d", "flag_tiny"]
    assert not current.loc["a", "flag_tiny"]
    closed = table[table["variant"] == "A_k1"].set_index("site_no")
    assert closed.loc["a", "area_20km_vs_current"] == 4.0


def test_broken_set_comes_from_the_current_rule():
    table = rules.wide_table(sample_long(), SETTINGS)
    assert rules.broken_sites(table) == {"a", "b"}


def test_smallest_qualifying_k_is_chosen():
    table = rules.wide_table(sample_long(), SETTINGS)
    choice = rules.choose_k(table, relaxed_settings(2))
    assert choice["chosen_k"] == 1
    assert choice["candidates"][0]["fixed"] == ["a", "b"]


def test_no_k_qualifies_when_too_few_sites_fixed():
    table = rules.wide_table(sample_long(), SETTINGS)
    assert rules.choose_k(table, SETTINGS)["chosen_k"] is None


def test_new_leakage_blocks_a_k():
    rows = sample_long()
    mask = (rows["variant"] == "A_k1") & (rows["site_no"] == "d") & (rows["roi"] == "extended")
    rows.loc[mask, "area_km2"] = 100.0
    table = rules.wide_table(rows, SETTINGS)
    choice = rules.choose_k(table, relaxed_settings(2))
    assert choice["candidates"][0]["new_leakage"] == ["d"]
    assert not choice["candidates"][0]["qualifies"]
    assert choice["chosen_k"] == 2


def test_cap_adopted_when_it_removes_leakage_only():
    table = rules.wide_table(sample_long(), SETTINGS)
    decision = rules.decide_cap(table, SETTINGS, 1)
    assert decision["adopt"]
    assert decision["leakage_before"] == ["c"]
    assert decision["leakage_after"] == []


def test_cap_rejected_when_it_shrinks_a_non_leaking_site():
    rows = sample_long()
    mask = (rows["variant"] == "B_k1") & (rows["site_no"] == "a") & (rows["roi"] == "extended")
    rows.loc[mask, "area_km2"] = 5.0
    decision = rules.decide_cap(rules.wide_table(rows, SETTINGS), SETTINGS, 1)
    assert not decision["adopt"]
    assert decision["non_leaking_sites_losing_area"] == ["a"]


def test_cap_not_adopted_without_chosen_k():
    table = rules.wide_table(sample_long(), SETTINGS)
    assert not rules.decide_cap(table, SETTINGS, None)["adopt"]


def river(width: int = 1, gap: int = 0) -> np.ndarray:
    water = np.zeros((7, 12), dtype=bool)
    water[3, :] = True
    if gap:
        water[3, 5 : 5 + gap] = False
    return water


def test_twin_component_without_closing_stops_at_gap():
    water = river(gap=1)
    seed = np.zeros_like(water)
    seed[3, 1] = True
    assert rules.twin_component(water, seed, 0).sum() == 5


def test_twin_component_with_closing_bridges_gap_and_adds_no_pixels():
    water = river(gap=1)
    seed = np.zeros_like(water)
    seed[3, 1] = True
    component = rules.twin_component(water, seed, 1)
    assert component.sum() == water.sum()
    assert not (component & ~water).any()


def test_twin_component_too_wide_gap_not_bridged():
    water = river(gap=3)
    seed = np.zeros_like(water)
    seed[3, 1] = True
    assert rules.twin_component(water, seed, 1).sum() == 5


def test_closing_does_not_join_separate_parallel_water():
    water = np.zeros((9, 9), dtype=bool)
    water[2, :] = True
    water[6, :] = True
    seed = np.zeros_like(water)
    seed[2, 0] = True
    assert rules.twin_component(water, seed, 1).sum() == 9


def test_variant_list_has_current_and_both_kinds():
    names = rules.variant_list(SETTINGS)
    assert names[0] == "current"
    assert "A_k3" in names and "B_k1" in names
