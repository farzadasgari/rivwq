from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

from rivwq.usgs import pool

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "site_freeze.yaml"
LAYER_ORDER = ["core", "extended_caveat", "extended_estimated"]
BOOLEAN_COLUMNS = [
    "pass_width_10px",
    "pass_turbidity_cov",
    "pass_overlap",
    "pass_overlap_conservative",
    "caveat",
    "coverage_estimated",
]
STRATA_COLUMNS = ["turbidity_class", "regulation_class", "regime_class", "cloud_class"]
TARGET_COLUMNS = {
    "turbidity": "target_turbidity_ok",
    "dissolved_oxygen": "target_do_ok",
    "fchl": "target_fchl_ok",
}


class FreezeError(RuntimeError):
    pass


class QuotedText(str):
    pass


class QuotingDumper(yaml.SafeDumper):
    pass


QuotingDumper.add_representer(
    QuotedText,
    lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:str", value, style="'"),
)


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def river_group(station: str, settings: dict) -> str:
    name = f" {station.upper()} "
    cuts = [name.find(f" {word} ") for word in settings["separators"]]
    cuts += [name.find(mark) for mark in settings["punctuation"]]
    positions = [cut for cut in cuts if cut > 0]
    head = name[: min(positions)] if positions else name
    tokens = head.split()
    if len(tokens) > 1 and tokens[-1] in settings["suffixes"]:
        tokens = tokens[:-1]
    slug = re.sub(r"[^a-z0-9]+", "_", " ".join(tokens).lower()).strip("_")
    return settings.get("aliases", {}).get(slug, slug)


def basin_group(huc8: str, digits: int) -> str:
    return huc8[:digits]


def layer_masks(table: pd.DataFrame) -> dict[str, pd.Series]:
    base = table["pass_width_10px"] & table["pass_turbidity_cov"]
    conservative = base & table["pass_overlap_conservative"]
    return {
        "core": conservative & ~table["caveat"] & ~table["coverage_estimated"],
        "extended_caveat": conservative & table["caveat"],
        "extended_estimated": base & table["pass_overlap"] & table["coverage_estimated"],
    }


def assign_layers(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    masks = layer_masks(table)
    membership = sum(mask.astype(int) for mask in masks.values())
    shared = list(table.index[membership > 1])
    if shared:
        raise FreezeError(f"sites in more than one layer: {shared}")
    return {name: table[masks[name]] for name in LAYER_ORDER}


def python_value(value: object) -> object:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def site_record(site: str, row: pd.Series, group: str, basin: str, fchl_code: str | None) -> dict:
    caveat_reason = python_value(row["caveat_reason"]) if row["caveat"] else None
    return {
        "site_no": QuotedText(site),
        "station": row["station"],
        "state": row["state"],
        "region": row["region"],
        "river_group": group,
        "basin_group": basin,
        "cv_group": None,
        "role": row["role"],
        "drainage_area_sqmi": python_value(row["drainage_area_sqmi"]),
        "pixels_across": python_value(row["pixels_across"]),
        "targets": {name: bool(row[column]) for name, column in TARGET_COLUMNS.items()},
        "fchl_code": QuotedText(fchl_code) if fchl_code else None,
        "caveat": bool(row["caveat"]),
        "caveat_reason": caveat_reason,
        "coverage_estimated": bool(row["coverage_estimated"]),
        "strata": {column: row[column] for column in STRATA_COLUMNS},
        "est_scenes_upper": python_value(row["est_scenes_upper"]),
        "est_scenes_matched": python_value(row["est_scenes_matched"]),
    }


def read_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"site": str}).set_index("site")
    table.index.name = "site_no"
    missing = [column for column in BOOLEAN_COLUMNS if column not in table.columns]
    if missing:
        raise FreezeError(f"table lacks columns: {missing}")
    for column in BOOLEAN_COLUMNS:
        if table[column].isna().any():
            raise FreezeError(f"missing values in rule column {column}")
        table[column] = table[column].astype(bool)
    return table


def read_site_attributes(paths_in_priority: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths_in_priority:
        frame = pd.read_csv(path, dtype=str).set_index("site_no")
        frames.append(frame[["huc8", "chlorophyll_fluorescence_code"]])
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.combine_first(frame[~frame.index.isin(combined.index)])
    return combined


def build_sites(
    table: pd.DataFrame, attributes: pd.DataFrame, settings: dict
) -> dict[str, list[dict]]:
    layers = assign_layers(table)
    members = [site for frame in layers.values() for site in frame.index]
    missing = [
        site
        for site in members
        if site not in attributes.index or pd.isna(attributes.loc[site, "huc8"])
    ]
    if missing:
        raise FreezeError(f"no HUC8 for sites: {missing}")
    result = {}
    for name, members in layers.items():
        records = []
        for site in sorted(members.index):
            row = members.loc[site]
            group = river_group(row["station"], settings["river_group"])
            basin = basin_group(attributes.loc[site, "huc8"], settings["basin_digits"])
            code = pool.normalise_code(attributes.loc[site, "chlorophyll_fluorescence_code"])
            records.append(site_record(site, row, group, basin, code))
        result[name] = records
    assign_cv_groups(result)
    return result


def grouping_conflicts(layers: dict[str, list[dict]]) -> dict[str, list[tuple[str, str]]]:
    records = [site for name in LAYER_ORDER for site in layers[name]]
    same_river: list[tuple[str, str]] = []
    same_basin: list[tuple[str, str]] = []
    for index, first in enumerate(records):
        for second in records[index + 1 :]:
            pair = (first["site_no"], second["site_no"])
            same_river_group = first["river_group"] == second["river_group"]
            same_basin_group = first["basin_group"] == second["basin_group"]
            if same_river_group and not same_basin_group:
                same_river.append(pair)
            if same_basin_group and not same_river_group:
                same_basin.append(pair)
    return {"same_river_different_basin": same_river, "same_basin_different_river": same_basin}


def connected_components(records: list[dict]) -> list[set[str]]:
    parent = {record["site_no"]: record["site_no"] for record in records}

    def find(site: str) -> str:
        while parent[site] != site:
            parent[site] = parent[parent[site]]
            site = parent[site]
        return site

    for key in ["river_group", "basin_group"]:
        first_by_value: dict[str, str] = {}
        for record in records:
            anchor = first_by_value.setdefault(record[key], record["site_no"])
            parent[find(record["site_no"])] = find(anchor)
    components: dict[str, set[str]] = {}
    for record in records:
        components.setdefault(find(record["site_no"]), set()).add(record["site_no"])
    return list(components.values())


def assign_cv_groups(layers: dict[str, list[dict]]) -> None:
    records = [site for name in LAYER_ORDER for site in layers[name]]
    by_site = {record["site_no"]: record for record in records}
    for members in connected_components(records):
        rivers = sorted({by_site[site]["river_group"] for site in members})
        label = "+".join(rivers)
        for site in members:
            by_site[site]["cv_group"] = label


def cv_group_summary(layers: dict[str, list[dict]], share_limit: float) -> dict[str, dict]:
    core_matched = sum(site["est_scenes_matched"] for site in layers["core"])
    summary: dict[str, dict] = {}
    for name in LAYER_ORDER:
        for site in layers[name]:
            entry = summary.setdefault(
                site["cv_group"],
                {
                    "members": [],
                    "sites_per_layer": dict.fromkeys(LAYER_ORDER, 0),
                    "est_scenes_upper": 0.0,
                    "est_scenes_matched": 0.0,
                    "core_est_scenes_matched": 0.0,
                },
            )
            entry["members"].append(site["site_no"])
            entry["sites_per_layer"][name] += 1
            entry["est_scenes_upper"] += site["est_scenes_upper"]
            entry["est_scenes_matched"] += site["est_scenes_matched"]
            if name == "core":
                entry["core_est_scenes_matched"] += site["est_scenes_matched"]
    for entry in summary.values():
        share = entry["core_est_scenes_matched"] / core_matched if core_matched else 0.0
        entry["core_scene_share"] = share
        entry["exceeds_core_scene_share_limit"] = share > share_limit
    return dict(sorted(summary.items()))


def layer_counts(layers: dict[str, list[dict]]) -> dict[str, int]:
    return {name: len(layers[name]) for name in LAYER_ORDER}


def scene_totals(layers: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "est_scenes_upper": sum(site["est_scenes_upper"] for site in layers[name]),
            "est_scenes_matched": sum(site["est_scenes_matched"] for site in layers[name]),
        }
        for name in LAYER_ORDER
    }


def freeze_document(
    layers: dict[str, list[dict]], source_name: str, source_sha256: str, share_limit: float
) -> dict:
    return {
        "source_table": source_name,
        "source_table_sha256": source_sha256,
        "counts": layer_counts(layers),
        "scene_totals": scene_totals(layers),
        "cv_groups": cv_group_summary(layers, share_limit),
        "grouping_conflicts": {
            key: [list(pair) for pair in pairs] for key, pairs in grouping_conflicts(layers).items()
        },
        "layers": layers,
    }


def check_counts(counts: dict[str, int], settings: dict, table_sha256: str) -> None:
    expected = settings["expected_counts"]
    if counts == expected:
        return
    if table_sha256 == settings["frozen_table_sha256"]:
        reason = "the table is unchanged, so the rules or the code changed the site set"
    else:
        reason = "the table changed; review it and update config/site_freeze.yaml deliberately"
    raise FreezeError(f"layer counts {counts} differ from expected {expected}: {reason}")


def render(document: dict) -> str:
    return yaml.dump(document, Dumper=QuotingDumper, sort_keys=False, allow_unicode=True, width=100)


def paths(settings: dict) -> tuple[Path, list[Path], Path]:
    input_dir = Path(settings["input_dir"]).expanduser()
    return (
        input_dir / settings["table_file"],
        [input_dir / name for name in settings["attribute_files"]],
        REPO_ROOT / settings["output_file"],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    settings = load_config(args.config)
    table_path, attribute_paths, output_path = paths(settings)
    table_sha256 = file_sha256(table_path)
    try:
        layers = build_sites(
            read_table(table_path), read_site_attributes(attribute_paths), settings
        )
    except FreezeError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    document = freeze_document(
        layers, table_path.name, table_sha256, settings["cv_group_core_scene_share_limit"]
    )
    try:
        check_counts(document["counts"], settings, table_sha256)
        if args.check_only and output_path.read_text(encoding="utf-8") != render(document):
            raise FreezeError(f"{output_path} is out of date; run make sites")
    except FreezeError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if not args.check_only:
        output_path.write_text(render(document), encoding="utf-8")
    for name in LAYER_ORDER:
        print(f"{name}: {document['counts'][name]}")
