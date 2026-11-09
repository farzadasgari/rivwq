from __future__ import annotations

import pandas as pd
import pytest
import yaml

from rivwq.gee import connectivity_rules as rules
from rivwq.gee import roi_flags

SETTINGS = rules.load_config()


def areas_for(site: str, near: float, reach: float, extended: float) -> list[dict]:
    rows = []
    for variant in rules.variant_list(SETTINGS):
        for name, area in (("near_gauge", near), ("reach", reach), ("extended", extended)):
            rows.append({"variant": variant, "site_no": site, "roi": name, "area_km2": area})
    return rows


def sample_areas() -> pd.DataFrame:
    rows = []
    rows += areas_for("ok", 0.3, 2.0, 8.0)
    rows += areas_for("broken", 0.3, 2.0, 2.2)
    rows += areas_for("leak", 0.3, 2.0, 20.0)
    rows += areas_for("tiny", 0.03, 0.06, 0.06)
    rows += areas_for("smallnear", 0.05, 1.0, 4.0)
    return pd.DataFrame(rows)


def build() -> dict:
    return roi_flags.build_flags(sample_areas(), SETTINGS, "areas.csv", "abc")


def test_each_criterion_flags_the_right_site():
    invalid = build()["extended_invalid"]
    assert invalid["broken"]["reasons"] == ["broken"]
    assert invalid["leak"]["reasons"] == ["leakage"]
    assert invalid["tiny"]["reasons"] == ["broken", "tiny_extended"]
    assert "ok" not in invalid


def test_small_near_gauge_alone_does_not_invalidate_the_extended_roi():
    flags = build()
    assert "smallnear" not in flags["extended_invalid"]
    assert "smallnear" in flags["near_gauge_tiny"]
    assert "ok" not in flags["near_gauge_tiny"]


def test_thresholds_are_read_from_config():
    flags = build()
    assert flags["criteria"] == SETTINGS["thresholds"]
    assert SETTINGS["thresholds"]["tiny_area_km2"] == 0.1


def test_counts_add_up():
    counts = build()["counts"]
    assert counts["extended_invalid"] == 3
    assert counts["broken"] == 2
    assert counts["leakage"] == 1
    assert counts["tiny_extended"] == 1


def test_only_the_current_rule_is_used():
    rows = sample_areas()
    mask = (rows["variant"] != "current") & (rows["site_no"] == "ok") & (rows["roi"] == "extended")
    rows.loc[mask, "area_km2"] = 2.0
    flags = roi_flags.build_flags(rows, SETTINGS, "areas.csv", "abc")
    assert "ok" not in flags["extended_invalid"]


def test_rendered_yaml_round_trips_with_string_site_numbers():
    flags = build()
    loaded = yaml.safe_load(roi_flags.render(flags))
    assert loaded == flags
    assert all(isinstance(site, str) for site in loaded["extended_invalid"])


def test_check_only_detects_a_stale_file(tmp_path, monkeypatch):
    source = tmp_path / "areas.csv"
    sample_areas().to_csv(source, index=False)
    monkeypatch.setattr(roi_flags.roi, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(roi_flags.rules.roi, "REPO_ROOT", tmp_path)
    settings = {**SETTINGS, "cache_dir": "."}
    monkeypatch.setattr(roi_flags.rules, "load_config", lambda: settings)
    output = tmp_path / "flags.yaml"
    roi_flags.main(["--output", str(output)])
    roi_flags.main(["--output", str(output), "--check-only"])
    output.write_text(output.read_text(encoding="utf-8") + "x: 1\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        roi_flags.main(["--output", str(output), "--check-only"])


def test_source_hash_is_recorded(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("a\n", encoding="utf-8")
    assert len(roi_flags.file_sha256(path)) == 64
