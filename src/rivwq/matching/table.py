from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rivwq.matching import readings, windows
from rivwq.usgs import qc_rules

SCENE_COLUMNS = ["system_index", "time_utc", "spacecraft", "mgrs_tile"]
ROI_KEY = ["roi", "erosion_px", "cloud_variant"]
ROI_VALUE_COLUMNS = ["total_count", "valid_count", "valid_fraction"]
PARAMETER_FIELDS = [
    "value",
    "n",
    "dt_min",
    "sensor_id",
    "approval",
    "provisional_share",
    "estimated_share",
    "native_min",
    "other_sensor_in_window",
]
APPROVAL_PROVISIONAL = "provisional"
APPROVAL_APPROVED = "approved"


def read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def site_layers(sites_config: dict) -> dict[str, dict]:
    result = {}
    for layer, members in sites_config["layers"].items():
        for member in members:
            result[str(member["site_no"])] = {**member, "layer": layer}
    return result


def invalid_extended_sites(flags_config: dict) -> dict[str, str]:
    return {
        str(site): "|".join(details["reasons"])
        for site, details in flags_config["extended_invalid"].items()
    }


def site_parameters(site: dict, config: dict) -> dict[str, str]:
    codes = dict(config["parameters"])
    codes["fchl"] = site.get("fchl_code")
    selected = {}
    for name in config["targets"]:
        if site["targets"].get(name) and codes.get(name):
            selected[name] = str(codes[name])
    for name in config["covariates"]:
        selected[name] = str(codes[name])
    return selected


def approval_label(n: pd.Series, provisional_share: pd.Series) -> pd.Series:
    label = pd.Series(pd.NA, index=n.index, dtype="object")
    label[(n > 0) & (provisional_share > 0)] = APPROVAL_PROVISIONAL
    label[(n > 0) & (provisional_share == 0)] = APPROVAL_APPROVED
    return label


def parameter_frame(name: str, matched: pd.DataFrame, matched_here: bool) -> pd.DataFrame:
    frame = matched.copy()
    frame["approval"] = approval_label(frame["n"], frame["provisional_share"])
    if not matched_here:
        for column in ("value", "dt_min", "provisional_share", "estimated_share", "native_min"):
            frame[column] = np.nan
        for column in ("sensor_id", "approval"):
            frame[column] = None
        frame["other_sensor_in_window"] = False
        frame["n"] = pd.array([pd.NA] * len(frame), dtype="Int64")
    frame = frame[["scene_pos", "variant", *PARAMETER_FIELDS]]
    return frame.rename(columns={field: f"{name}_{field}" for field in PARAMETER_FIELDS})


def all_parameter_names(config: dict) -> list[str]:
    return list(config["targets"]) + list(config["covariates"])


def scene_matches(
    tidy: pd.DataFrame,
    scenes: pd.DataFrame,
    parameters: dict[str, str],
    config: dict,
    qc_config: dict,
    level: str,
) -> tuple[pd.DataFrame, dict[str, list[readings.ReadingSeries]]]:
    scenes = scenes.reset_index(drop=True)
    variants = windows.window_variants(config)
    base = pd.DataFrame(
        {
            "scene_pos": np.repeat(np.arange(len(scenes)), len(variants)),
            "variant": np.tile(variants, len(scenes)),
        }
    )
    series_by_name = {}
    for name in all_parameter_names(config):
        if name in parameters:
            series = readings.build_series(tidy, config, qc_config, level, parameters[name])
            matched = windows.match_parameter(series, scenes["time_utc"], config)
            series_by_name[name] = series
            part = parameter_frame(name, matched, True)
        else:
            empty = windows.match_parameter([], scenes["time_utc"], config)
            part = parameter_frame(name, empty, False)
        base = base.merge(part, on=["scene_pos", "variant"], how="left")
    scene_part = scenes[SCENE_COLUMNS].copy()
    scene_part.insert(0, "scene_pos", np.arange(len(scenes)))
    base = scene_part.merge(base, on="scene_pos", how="right")
    base.insert(0, "site_no", scenes["site_no"].iloc[0] if len(scenes) else None)
    base["qc_level"] = level
    return base.drop(columns="scene_pos"), series_by_name


def unique_scenes(gee: pd.DataFrame) -> pd.DataFrame:
    columns = ["site_no", *SCENE_COLUMNS]
    return (
        gee[columns]
        .drop_duplicates("system_index")
        .sort_values(["time_utc", "system_index"])
        .reset_index(drop=True)
    )


def expand_to_rois(
    matches: pd.DataFrame,
    gee: pd.DataFrame,
    site: dict,
    invalid_extended: dict[str, str],
    config: dict,
) -> pd.DataFrame:
    roi_rows = gee[["system_index", *ROI_KEY, *ROI_VALUE_COLUMNS]]
    table = matches.merge(roi_rows, on="system_index", how="inner")
    table["site_layer"] = site["layer"]
    table["cv_group"] = site["cv_group"]
    table["sensitivity_only"] = site["site_no"] in config["sensitivity_sites"]
    invalid = table["site_no"].isin(invalid_extended) & (table["roi"] == "extended")
    table["extended_roi_valid"] = ~invalid
    table["roi_flag_reason"] = np.where(invalid, table["site_no"].map(invalid_extended), "")
    return table.sort_values(
        ["time_utc", "system_index", "roi", "erosion_px", "cloud_variant", "variant"]
    ).reset_index(drop=True)


def empty_window_counts(matches: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for name in all_parameter_names(config):
        n_column = f"{name}_n"
        part = matches[matches[n_column].notna()]
        if part.empty:
            continue
        grouped = part.groupby(["site_no", "variant"])
        for (site, variant), group in grouped:
            empty = int((group[n_column] == 0).sum())
            scenes = len(group)
            rows.append(
                {
                    "site_no": site,
                    "parameter": name,
                    "variant": variant,
                    "scenes": scenes,
                    "empty_windows": empty,
                    "empty_share": empty / scenes,
                    "mostly_empty": empty / scenes > config["mostly_empty_share"],
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "site_no",
            "parameter",
            "variant",
            "scenes",
            "empty_windows",
            "empty_share",
            "mostly_empty",
        ],
    )


def estimated_share_table(
    series_by_name: dict[str, list[readings.ReadingSeries]], matches: pd.DataFrame, config: dict
) -> pd.DataFrame:
    rows = []
    for name in config["covariates"]:
        series = series_by_name.get(name, [])
        total = sum(len(item.values) for item in series)
        estimated = sum(int(item.estimated.sum()) for item in series)
        nearest = matches[(matches["variant"] == windows.NEAREST) & matches[f"{name}_n"].gt(0)]
        site = matches["site_no"].iloc[0] if len(matches) else None
        rows.append(
            {
                "site_no": site,
                "parameter": name,
                "readings": total,
                "estimated_readings": estimated,
                "estimated_share_readings": estimated / total if total else np.nan,
                "scenes_with_nearest": len(nearest),
                "scenes_nearest_estimated": int((nearest[f"{name}_estimated_share"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def exclusion_counts(
    tidy: pd.DataFrame, parameters: dict[str, str], qc_config: dict, level: str
) -> pd.DataFrame:
    selected = tidy[tidy["parameter_code"].isin(parameters.values())]
    if selected.empty:
        return pd.DataFrame(
            columns=["level", "site_no", "parameter_code", "rule", "rows", "flagged"]
        )
    flags = qc_rules.apply_level(selected, qc_config, level)
    return qc_rules.count_flags(flags, level)


def build_site(
    tidy: pd.DataFrame,
    gee: pd.DataFrame,
    site: dict,
    invalid_extended: dict[str, str],
    config: dict,
    qc_config: dict,
    level: str,
) -> dict[str, pd.DataFrame]:
    parameters = site_parameters(site, config)
    scenes = unique_scenes(gee)
    matches, series_by_name = scene_matches(tidy, scenes, parameters, config, qc_config, level)
    return {
        "scene_matches": matches,
        "table": expand_to_rois(matches, gee, site, invalid_extended, config),
        "empty_windows": empty_window_counts(matches, config),
        "estimated": estimated_share_table(series_by_name, matches, config),
        "exclusions": exclusion_counts(tidy, parameters, qc_config, level),
    }


def read_gee_site(run_dir: Path, site_no: str) -> pd.DataFrame:
    columns = ["site_no", *SCENE_COLUMNS, *ROI_KEY, *ROI_VALUE_COLUMNS]
    paths = sorted((run_dir / "tidy" / site_no).glob("*.parquet"))
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def gee_sites(run_dir: Path) -> list[str]:
    return sorted(path.name for path in (run_dir / "tidy").iterdir() if path.is_dir())


def run(
    config: dict,
    gee_run_id: str,
    levels: list[str],
    only_sites: list[str] | None,
    repo_root: Path,
) -> Path:
    qc_config = qc_rules.load_config()
    sites = site_layers(read_yaml(repo_root / config["sites_file"]))
    invalid = invalid_extended_sites(read_yaml(repo_root / config["roi_flags_file"]))
    gee_dir = repo_root / config["gee_root"] / gee_run_id
    usgs_dir = repo_root / config["usgs_run_dir"]
    out_dir = repo_root / config["output_root"] / gee_run_id
    selected = only_sites or gee_sites(gee_dir)
    collected: dict[str, list[pd.DataFrame]] = {}
    for site_no in selected:
        gee = read_gee_site(gee_dir, site_no)
        if gee.empty:
            continue
        tidy = qc_rules.read_site(usgs_dir, site_no)
        for level in levels:
            parts = build_site(tidy, gee, sites[site_no], invalid, config, qc_config, level)
            target = out_dir / level
            target.mkdir(parents=True, exist_ok=True)
            parts["table"].to_parquet(target / f"matched_{site_no}.parquet", index=False)
            parts["scene_matches"].to_parquet(target / f"scenes_{site_no}.parquet", index=False)
            for name in ("empty_windows", "estimated", "exclusions"):
                collected.setdefault(f"{level}/{name}", []).append(parts[name])
    for key, frames in collected.items():
        level, name = key.split("/")
        pd.concat(frames, ignore_index=True).to_csv(out_dir / level / f"{name}.csv", index=False)
    return out_dir
