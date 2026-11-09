from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import yaml

from rivwq.gee import connectivity_rules as rules
from rivwq.gee import roi

FLAGS_PATH = roi.REPO_ROOT / "config" / "roi_flags.yaml"
TEXT = {"site_no": str}
ROUNDING = 4


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reasons_for(row: pd.Series, limits: dict) -> list[str]:
    reasons = []
    if row["ratio_20_5"] < limits["broken_ratio"]:
        reasons.append("broken")
    if row["ratio_20_5"] > limits["leakage_ratio"]:
        reasons.append("leakage")
    if row["area_20km"] < limits["tiny_area_km2"]:
        reasons.append("tiny_extended")
    return reasons


def classify(table: pd.DataFrame, settings: dict) -> dict:
    current = table[table["variant"] == rules.CURRENT].set_index("site_no").sort_index()
    limits = settings["thresholds"]
    invalid = {}
    for site, row in current.iterrows():
        reasons = reasons_for(row, limits)
        if reasons:
            invalid[str(site)] = {
                "reasons": reasons,
                "area_1km_km2": round(float(row["area_1km"]), ROUNDING),
                "area_5km_km2": round(float(row["area_5km"]), ROUNDING),
                "area_20km_km2": round(float(row["area_20km"]), ROUNDING),
                "ratio_20_5": round(float(row["ratio_20_5"]), ROUNDING),
            }
    tiny_near = sorted(
        str(site) for site in current.index[current["area_1km"] < limits["tiny_area_km2"]]
    )
    return {
        "extended_invalid": invalid,
        "near_gauge_tiny": tiny_near,
        "sites_surveyed": len(current),
    }


def build_flags(areas: pd.DataFrame, settings: dict, source: str, source_sha256: str) -> dict:
    table = rules.wide_table(areas, settings)
    result = classify(table, settings)
    counts = {}
    for entry in result["extended_invalid"].values():
        for reason in entry["reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    return {
        "source_table": source,
        "source_table_sha256": source_sha256,
        "variant": rules.CURRENT,
        "applies_to": "analysis time only; raw statistics are stored for every ROI",
        "criteria": dict(settings["thresholds"]),
        "counts": {
            "extended_invalid": len(result["extended_invalid"]),
            **dict(sorted(counts.items())),
        },
        "extended_invalid": result["extended_invalid"],
        "near_gauge_tiny": result["near_gauge_tiny"],
    }


def render(flags: dict) -> str:
    return yaml.safe_dump(flags, sort_keys=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, default=FLAGS_PATH)
    args = parser.parse_args(argv)
    settings = rules.load_config()
    source = roi.REPO_ROOT / settings["cache_dir"] / "areas.csv"
    areas = pd.read_csv(source, dtype=TEXT)
    flags = build_flags(areas, settings, settings["cache_dir"] + "/areas.csv", file_sha256(source))
    text = render(flags)
    if args.check_only:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != text:
            raise SystemExit("config/roi_flags.yaml differs from the survey table")
        print("roi flags up to date")
        return
    args.output.write_text(text, encoding="utf-8")
    print(f"extended invalid: {flags['counts']['extended_invalid']}")
