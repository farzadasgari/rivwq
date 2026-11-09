from __future__ import annotations

import pytest
import yaml

from rivwq.gee import roi, roi_plan

CONFIG = yaml.safe_load((roi.REPO_ROOT / "config" / "gee.yaml").read_text(encoding="utf-8"))
SITES = sorted(roi.load_gauges(roi.REPO_ROOT / CONFIG["roi"]["gauges_file"]))


def flags_for(tmp_path, sites: list[str]) -> dict:
    path = tmp_path / "flags.yaml"
    path.write_text(
        yaml.safe_dump({"extended_invalid": {site: {"reasons": ["broken"]} for site in sites}}),
        encoding="utf-8",
    )
    return {**CONFIG, "roi_plan": {**CONFIG["roi_plan"], "flags_file": str(path)}}


def test_skip_list_is_read_from_the_generated_flags_file():
    skips = roi_plan.skips_from_flags(CONFIG, SITES)
    generated = yaml.safe_load(roi_plan.flags_path(CONFIG).read_text(encoding="utf-8"))
    assert sorted(skips) == sorted(str(site) for site in generated["extended_invalid"])
    assert len(skips) == 10
    assert all(names == ["extended"] for names in skips.values())


def test_flagged_sites_lose_only_the_extended_roi():
    skips = roi_plan.skips_from_flags(CONFIG, SITES)
    assert roi_plan.kept_rois(CONFIG, skips, "06887500") == ["near_gauge", "reach"]
    assert roi_plan.kept_rois(CONFIG, skips, "03378500") == ["near_gauge", "reach", "extended"]


def test_counts_for_the_frozen_sites():
    skips = roi_plan.skips_from_flags(CONFIG, SITES)
    counts = roi_plan.combination_counts(roi_plan.site_plan(CONFIG, SITES, skips), CONFIG)
    assert counts == {
        "sites": 24,
        "site_roi_possible": 72,
        "site_roi_planned": 62,
        "site_roi_skipped": 10,
    }


def test_empty_skips_keep_everything():
    plan = roi_plan.site_plan(CONFIG, SITES, {})
    assert all(names == list(CONFIG["roi"]["variants"]) for names in plan.values())
    assert roi_plan.combination_counts(plan, CONFIG)["site_roi_skipped"] == 0


def test_unknown_flagged_site_is_refused(tmp_path):
    config = flags_for(tmp_path, ["nope"])
    with pytest.raises(ValueError):
        roi_plan.skips_from_flags(config, SITES)


def test_skipping_an_unconfigured_roi_is_refused():
    config = {**CONFIG, "roi_plan": {**CONFIG["roi_plan"], "skipped_roi": "missing"}}
    with pytest.raises(ValueError):
        roi_plan.skips_from_flags(config, SITES)


def test_the_skip_list_follows_the_flags_file(tmp_path):
    config = flags_for(tmp_path, [SITES[0], SITES[1]])
    assert sorted(roi_plan.skips_from_flags(config, SITES)) == sorted([SITES[0], SITES[1]])


def test_flags_hash_is_a_sha256_of_the_file():
    assert len(roi_plan.flags_sha256(CONFIG)) == 64
