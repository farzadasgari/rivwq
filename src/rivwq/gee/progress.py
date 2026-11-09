from __future__ import annotations

from rivwq.gee import status, years


def year_of(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def site_of(name: str) -> str:
    return name.rsplit("_", 1)[0]


def year_factor(year: int, eta: dict) -> float:
    return float(eta["scene_factor_by_year"].get(year, 1.0))


def seconds_per_scene(manifest: dict, eta: dict) -> float:
    timed = [item for item in manifest["files"] if item.get("seconds") and item.get("scenes")]
    scenes = sum(item["scenes"] for item in timed)
    if not scenes:
        return float(eta["seconds_per_scene"])
    return sum(item["seconds"] for item in timed) / scenes


def scenes_by_site(manifest: dict, eta: dict) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for item in manifest["files"]:
        if not item.get("scenes") or "_" not in item["path"]:
            continue
        name = item["path"].rsplit("/", 1)[-1].removesuffix(".parquet")
        factor = year_factor(year_of(name), eta)
        totals.setdefault(site_of(name), []).append(item["scenes"] / factor)
    return {site: sum(values) / len(values) for site, values in totals.items()}


def expected_scenes(site: str, year: int, per_site: dict[str, float], eta: dict) -> float:
    base = per_site.get(site, float(eta["scenes_per_site_year"]))
    return base * year_factor(year, eta)


def remaining_hours(pending: list[str], manifest: dict, config: dict) -> float:
    eta = config["eta"]
    per_site = scenes_by_site(manifest, eta)
    rate = seconds_per_scene(manifest, eta)
    seconds = sum(
        expected_scenes(site_of(name), year_of(name), per_site, eta) * rate for name in pending
    )
    return seconds / max(config["workers"], 1) / 3600.0


def phase_years(phase: dict, first_year: int, last_year: int) -> list[int]:
    return years.parse_years(str(phase["years"]), first_year, last_year)


def universe_names(sites: list[str], year_list: list[int]) -> list[str]:
    return [f"{site}_{year}" for site in sites for year in year_list]


def group_progress(names: list[str], done: set[str], failed: dict, manifest: dict, config: dict):
    open_names = [name for name in names if name not in done]
    return {
        "total": len(names),
        "done": len(names) - len(open_names),
        "pending": len(open_names) - sum(1 for name in open_names if name in failed),
        "failed": sum(1 for name in open_names if name in failed),
        "eta_hours": remaining_hours(open_names, manifest, config),
    }


def report(
    state: dict,
    manifest: dict,
    config: dict,
    alive: bool,
    selected_years: list[int] | None = None,
) -> dict:
    first_year = int(config["date_start"][:4])
    last_year = max(
        [year_of(name) for name in state["done"] + state["pending"] + list(state["failed"])]
        + [first_year]
        + [year for phase in config["phases"] for year in phase_years(phase, first_year, 9999)]
    )
    sites = list(manifest["sites"])
    done = set(state["done"])
    failed = state["failed"]
    configured = [phase_years(phase, first_year, last_year) for phase in config["phases"]]
    seen = {year_of(name) for name in state["done"] + state["pending"] + list(failed)}
    all_years = sorted({year for group in configured for year in group} | seen)
    if selected_years is not None:
        all_years = [year for year in all_years if year in selected_years]
    by_year = {
        year: group_progress(universe_names(sites, [year]), done, failed, manifest, config)
        for year in all_years
    }
    by_phase = []
    for phase, group in zip(config["phases"], configured, strict=True):
        chosen = [year for year in group if selected_years is None or year in selected_years]
        if not chosen:
            continue
        entry = group_progress(universe_names(sites, chosen), done, failed, manifest, config)
        by_phase.append({"name": phase["name"], "years": chosen, **entry})
    universe = universe_names(sites, all_years)
    total = group_progress(universe, done, failed, manifest, config)
    return {
        "summary": status.summarise(state, manifest, config["workers"], alive),
        "seconds_per_scene": seconds_per_scene(manifest, config["eta"]),
        "workers": config["workers"],
        "overall": total,
        "by_phase": by_phase,
        "by_year": by_year,
    }


def format_report(result: dict) -> str:
    lines = [
        f"run {result['summary']['run_id']}  process_alive={result['summary']['process_alive']}"
        f"  workers={result['workers']}  s/scene={result['seconds_per_scene']:.1f}",
        f"overall: {result['overall']['done']}/{result['overall']['total']} site-years done,"
        f" {result['overall']['failed']} failed, about {result['overall']['eta_hours']:.1f} h left",
    ]
    for phase in result["by_phase"]:
        first, last = phase["years"][0], phase["years"][-1]
        span = str(first) if first == last else f"{first}-{last}"
        lines.append(
            f"  {phase['name']} {span}: {phase['done']}/{phase['total']} done,"
            f" {phase['failed']} failed, about {phase['eta_hours']:.1f} h left"
        )
    for year, entry in result["by_year"].items():
        lines.append(
            f"    {year}: {entry['done']}/{entry['total']} done, {entry['failed']} failed,"
            f" about {entry['eta_hours']:.1f} h left"
        )
    return "\n".join(lines)
