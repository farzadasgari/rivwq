from __future__ import annotations

import pandas as pd

from rivwq.gee import connectivity_rules as rules
from rivwq.gee import connectivity_survey as survey

SETTINGS = rules.load_config()


def test_variant_parameters_follow_config():
    assert survey.variant_parameters("current", SETTINGS) == (0, SETTINGS["current_path_m"])
    assert survey.variant_parameters("A_k2", SETTINGS) == (2, SETTINGS["current_path_m"])
    assert survey.variant_parameters("B_k3", SETTINGS) == (3, SETTINGS["cap_path_m"])


def test_plot_is_written(tmp_path):
    rows = []
    for variant in rules.variant_list(SETTINGS):
        for site, ratio in (("a", 1.0), ("b", 3.0)):
            for roi_name, area in (("near_gauge", 1.0), ("reach", 2.0), ("extended", 2.0 * ratio)):
                rows.append(
                    {"variant": variant, "site_no": site, "roi": roi_name, "area_km2": area}
                )
    table = rules.wide_table(pd.DataFrame(rows), SETTINGS)
    path = tmp_path / "plot.png"
    survey.plot_ratios(table, SETTINGS, {"a": "alpha"}, path)
    assert path.stat().st_size > 1000
