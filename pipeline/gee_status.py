import argparse
import json

from rivwq.gee import export, manifest, progress, roi, status, years


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--years")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    config = export.load_config()
    run_dir = roi.REPO_ROOT / config["raw_root"] / args.run_id
    state = manifest.read_json(run_dir / "run_state.json")
    current = manifest.read_json(run_dir / "manifest.json")
    pid_path = roi.REPO_ROOT / config["log_dir"] / f"{args.run_id}.pid"
    lock_path = run_dir / export.LOCK_NAME
    alive = status.pid_alive(pid_path) or status.pid_alive(lock_path)
    selected = None
    if args.years:
        first_year = int(config["date_start"][:4])
        selected = years.parse_years(args.years, first_year, 9999)
    result = progress.report(state, current, config, alive, selected)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(progress.format_report(result))
    for name, message in state["failed"].items():
        print(f"FAILED {name}: {message}")


if __name__ == "__main__":
    main()
