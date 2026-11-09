from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from rivwq.matching import audits
from rivwq.usgs import qc_rules

ROI_COLUMNS = ["roi", "erosion_px", "cloud_variant"]
VALUE_ANY = 0.0


class IncompleteRunError(RuntimeError):
    pass


def require_complete_run(run_dir: Path) -> None:
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    if state["pending"] or state["failed"]:
        raise IncompleteRunError(
            f"{state['run_id']}: {len(state['pending'])} pending, {len(state['failed'])} failed"
        )


def target_names(config: dict) -> list[str]:
    return list(config["targets"])


def utc_dates(times: pd.Series) -> pd.Series:
    return times.dt.tz_convert("UTC").dt.floor("D")


def effective_n(dates: pd.Series, values: pd.Series, settings: dict) -> dict:
    frame = pd.DataFrame({"date": dates.to_numpy(), "value": values.to_numpy()}).dropna()
    daily = frame.groupby("date")["value"].mean().sort_index()
    count = len(daily)
    result = {"dates": count, "pairs": 0, "autocorrelation": np.nan, "n_eff": np.nan}
    if count < 3:
        return result
    gaps = daily.index.to_series().diff().dt.total_seconds() / 86400.0
    close = (gaps <= settings["autocorrelation_max_gap_days"]).to_numpy()[1:]
    current = daily.to_numpy()[1:][close]
    previous = daily.to_numpy()[:-1][close]
    result["pairs"] = int(close.sum())
    if result["pairs"] < 3 or np.std(current) == 0 or np.std(previous) == 0:
        return result
    correlation = float(np.corrcoef(previous, current)[0, 1])
    clipped = min(max(correlation, 0.0), settings["autocorrelation_clip"])
    result["autocorrelation"] = correlation
    result["n_eff"] = count * (1.0 - clipped) / (1.0 + clipped)
    return result


def scene_level(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop_duplicates(["system_index", *ROI_COLUMNS, "variant"])


def funnel_rows(site_table: pd.DataFrame, config: dict) -> pd.DataFrame:
    settings = config["gate_a"]
    rows = []
    keys = [*ROI_COLUMNS, "variant"]
    for key, group in site_table.groupby(keys, sort=True):
        for level in settings["valid_fraction_levels"]:
            masked = (
                group[group["valid_count"] > 0]
                if level == VALUE_ANY
                else group[group["valid_fraction"] >= level]
            )
            row = dict(zip(keys, key, strict=True))
            row.update(
                {
                    "site_no": group["site_no"].iloc[0],
                    "valid_fraction_min": level,
                    "scenes_total": group["system_index"].nunique(),
                    "scenes_after_masks": masked["system_index"].nunique(),
                    "median_valid_fraction": float(group["valid_fraction"].median()),
                }
            )
            for name in target_names(config):
                column = f"{name}_n"
                with_target = masked[masked[column].fillna(0) > 0]
                row[f"{name}_scenes"] = with_target["system_index"].nunique()
                result = effective_n(
                    utc_dates(with_target["time_utc"]), with_target[f"{name}_value"], settings
                )
                row[f"{name}_dates"] = result["dates"]
                row[f"{name}_autocorrelation"] = result["autocorrelation"]
                row[f"{name}_n_eff"] = result["n_eff"]
            rows.append(row)
    return pd.DataFrame(rows)


def all_day_turbidity(tidy: pd.DataFrame, flags: pd.DataFrame, code: str) -> pd.Series:
    kept = audits.kept_series(tidy, flags, code)
    if kept.empty:
        return pd.Series(dtype="float64")
    return audits.daily_mean(kept["datetime_utc"], kept["value"]).sort_index()


def decile_shares(all_days: pd.Series, matched_days: pd.Series, bins: int) -> pd.DataFrame:
    edges = np.quantile(all_days.to_numpy(), np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    counts_all = np.histogram(all_days.to_numpy(), bins=edges)[0]
    counts_matched = np.histogram(matched_days.to_numpy(), bins=edges)[0]
    return pd.DataFrame(
        {
            "decile": np.arange(1, len(counts_all) + 1),
            "all_days": counts_all,
            "matched_days": counts_matched,
            "all_share": counts_all / max(counts_all.sum(), 1),
            "matched_share": counts_matched / max(counts_matched.sum(), 1),
        }
    )


def selection_stages(site_table: pd.DataFrame, config: dict) -> dict[str, pd.Series]:
    settings = config["gate_a"]
    headline = site_table[
        (site_table["roi"] == settings["roi_headline"])
        & (site_table["erosion_px"] == settings["erosion_headline"])
        & (site_table["cloud_variant"] == settings["cloud_headline"])
        & (site_table["variant"] == "nearest")
    ]
    masked = headline[headline["valid_count"] > 0]
    return {
        "all_scene_dates": utc_dates(headline["time_utc"]).drop_duplicates(),
        "after_masks": utc_dates(masked["time_utc"]).drop_duplicates(),
        "after_masks_with_turbidity": utc_dates(
            masked[masked["turbidity_n"].fillna(0) > 0]["time_utc"]
        ).drop_duplicates(),
    }


def selection_bias(
    site_table: pd.DataFrame, all_days: pd.Series, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    site = site_table["site_no"].iloc[0]
    shares, summary = [], []
    for stage, dates in selection_stages(site_table, config).items():
        matched = all_days.reindex(dates.to_numpy()).dropna()
        row = {
            "site_no": site,
            "stage": stage,
            "dates": len(dates),
            "dates_without_turbidity": len(dates) - len(matched),
            "all_days": len(all_days),
            "matched_days": len(matched),
            "ks_statistic": np.nan,
            "median_all": float(all_days.median()) if len(all_days) else np.nan,
            "median_matched": float(matched.median()) if len(matched) else np.nan,
        }
        if len(all_days) and len(matched):
            row["ks_statistic"] = float(stats.ks_2samp(all_days.to_numpy(), matched.to_numpy())[0])
            part = decile_shares(all_days, matched, config["gate_a"]["quantiles"])
            part.insert(0, "stage", stage)
            part.insert(0, "site_no", site)
            shares.append(part)
        summary.append(row)
    shares_frame = pd.concat(shares, ignore_index=True) if shares else pd.DataFrame()
    return shares_frame, pd.DataFrame(summary)


def read_matched(level_dir: Path, site_no: str) -> pd.DataFrame:
    return pd.read_parquet(level_dir / f"matched_{site_no}.parquet")


def build_report(
    level_dir: Path, usgs_dir: Path, config: dict, only_sites: list[str] | None
) -> dict[str, pd.DataFrame]:
    qc_config = qc_rules.load_config()
    sites = sorted(
        only_sites or [path.stem.removeprefix("matched_") for path in level_dir.glob("matched_*")]
    )
    funnels, shares, summaries = [], [], []
    code = config["parameters"]["turbidity"]
    for site_no in sites:
        site_table = read_matched(level_dir, site_no)
        funnels.append(funnel_rows(site_table, config))
        tidy = qc_rules.read_site(usgs_dir, site_no)
        subset = tidy[tidy["parameter_code"] == code]
        flags = qc_rules.apply_level(subset, qc_config, site_table["qc_level"].iloc[0])
        share, summary = selection_bias(site_table, all_day_turbidity(subset, flags, code), config)
        shares.append(share)
        summaries.append(summary)
    return {
        "gate_a_funnel": pd.concat(funnels, ignore_index=True),
        "gate_a_selection_deciles": pd.concat(shares, ignore_index=True),
        "gate_a_selection_summary": pd.concat(summaries, ignore_index=True),
    }


def run(
    config: dict, gee_run_id: str, level: str, only_sites: list[str] | None, repo_root: Path
) -> dict[str, pd.DataFrame]:
    require_complete_run(repo_root / config["gee_root"] / gee_run_id)
    level_dir = repo_root / config["output_root"] / gee_run_id / level
    usgs_dir = repo_root / config["usgs_run_dir"]
    return build_report(level_dir, usgs_dir, config, only_sites)


def write(result: dict[str, pd.DataFrame], report_dir: Path, prefix: str) -> list[Path]:
    paths = []
    for name, frame in result.items():
        path = report_dir / f"{prefix}{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths
