from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from rivwq.gee import roi


def flags_path(config: dict) -> Path:
    return roi.REPO_ROOT / config["roi_plan"]["flags_file"]


def flags_sha256(config: dict) -> str:
    return hashlib.sha256(flags_path(config).read_bytes()).hexdigest()


def skips_from_flags(config: dict, known_sites: list[str]) -> dict[str, list[str]]:
    settings = config["roi_plan"]
    skipped = settings["skipped_roi"]
    if skipped not in config["roi"]["variants"]:
        raise ValueError(f"skipped ROI {skipped} is not a configured ROI variant")
    flags = yaml.safe_load(flags_path(config).read_text(encoding="utf-8"))
    flagged = sorted(str(site) for site in flags[settings["flags_key"]])
    unknown = [site for site in flagged if site not in known_sites]
    if unknown:
        raise ValueError(f"flagged sites are not in the site list: {unknown}")
    return {site: [skipped] for site in flagged}


def kept_rois(config: dict, skips: dict[str, list[str]], site: str) -> list[str]:
    removed = set(skips.get(site, []))
    return [name for name in config["roi"]["variants"] if name not in removed]


def site_plan(config: dict, sites: list[str], skips: dict[str, list[str]]) -> dict[str, list[str]]:
    return {site: kept_rois(config, skips, site) for site in sites}


def combination_counts(plan: dict[str, list[str]], config: dict) -> dict:
    possible = len(plan) * len(config["roi"]["variants"])
    planned = sum(len(names) for names in plan.values())
    return {
        "sites": len(plan),
        "site_roi_possible": possible,
        "site_roi_planned": planned,
        "site_roi_skipped": possible - planned,
    }
