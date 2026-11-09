from __future__ import annotations

from pathlib import Path

import pandas as pd

from rivwq.gee import progress


def load_dates_per_year(path: Path) -> dict[str, float]:
    table = pd.read_csv(path, dtype={"site": str}).set_index("site")
    return {site: float(value) for site, value in table["dates_per_year"].items()}


def scenes_for(site: str, year: int, dates: dict[str, float], eta: dict) -> float:
    return eta["scenes_per_date"] * dates[site] * progress.year_factor(year, eta)


def phase_hours(
    config: dict, plan: dict[str, list[str]], dates: dict[str, float], last_year: int
) -> list[dict]:
    eta = config["eta"]
    first_year = int(config["date_start"][:4])
    rows = []
    for phase in config["phases"]:
        years = progress.phase_years(phase, first_year, last_year)
        seconds = 0.0
        scenes = 0.0
        for site in plan:
            weight = progress.cost_weight(site, plan, config)
            for year in years:
                count = scenes_for(site, year, dates, eta)
                scenes += count
                seconds += count * eta["seconds_per_scene"] * weight
        rows.append(
            {
                "name": phase["name"],
                "years": years,
                "scenes": scenes,
                "hours": seconds / max(config["workers"], 1) / 3600.0,
            }
        )
    return rows
