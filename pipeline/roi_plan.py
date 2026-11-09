import argparse
import json
from datetime import UTC, datetime

from rivwq.gee import estimate, export, roi, roi_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    config = export.load_config()
    gauges = roi.load_gauges(roi.REPO_ROOT / config["roi"]["gauges_file"])
    sites = sorted(gauges)
    skips = roi_plan.skips_from_flags(config, sites)
    plan = roi_plan.site_plan(config, sites, skips)
    full_plan = roi_plan.site_plan(config, sites, {})
    dates = estimate.load_dates_per_year(roi.REPO_ROOT / config["eta"]["dates_table"])
    last_year = datetime.now(UTC).year
    result = {
        "flags_file": config["roi_plan"]["flags_file"],
        "flags_sha256": roi_plan.flags_sha256(config),
        "skipped_site_rois": sorted(
            f"{site}:{name}" for site, names in skips.items() for name in names
        ),
        "roi_hash_without_skips": roi.roi_hash(gauges, config),
        "roi_hash_with_skips": roi.roi_hash(gauges, config, skips),
        "combinations": roi_plan.combination_counts(plan, config),
        "hours_per_phase_all_rois": estimate.phase_hours(config, full_plan, dates, last_year),
        "hours_per_phase_with_skips": estimate.phase_hours(config, plan, dates, last_year),
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"flags file: {result['flags_file']} (sha256 {result['flags_sha256'][:12]})")
    print(f"skipped site-ROI combinations: {', '.join(result['skipped_site_rois'])}")
    print(f"roi hash without skips: {result['roi_hash_without_skips']}")
    print(f"roi hash with skips:    {result['roi_hash_with_skips']}")
    counts = result["combinations"]
    print(
        f"sites {counts['sites']}, site-ROI combinations {counts['site_roi_planned']} planned of"
        f" {counts['site_roi_possible']} ({counts['site_roi_skipped']} skipped)"
    )
    print("phase     years        scenes   hours (all ROIs)   hours (with skips)")
    full_rows = result["hours_per_phase_all_rois"]
    cut_rows = result["hours_per_phase_with_skips"]
    for full, cut in zip(full_rows, cut_rows, strict=True):
        first, last = full["years"][0], full["years"][-1]
        span = str(first) if first == last else f"{first}-{last}"
        row = f"{full['name']:9} {span:11} {full['scenes']:8.0f}"
        print(f"{row} {full['hours']:12.1f} {cut['hours']:18.1f}")
    print(
        f"total {sum(r['scenes'] for r in cut_rows):17.0f}"
        f" {sum(r['hours'] for r in full_rows):12.1f} {sum(r['hours'] for r in cut_rows):18.1f}"
    )


if __name__ == "__main__":
    main()
