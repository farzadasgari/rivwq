import argparse
import json

from rivwq.gee import export, manifest, roi, status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = export.load_config()
    run_dir = roi.REPO_ROOT / config["raw_root"] / args.run_id
    state = manifest.read_json(run_dir / "run_state.json")
    current = manifest.read_json(run_dir / "manifest.json")
    pid_path = roi.REPO_ROOT / config["log_dir"] / f"{args.run_id}.pid"
    summary = status.summarise(state, current, config["workers"], status.pid_alive(pid_path))
    print(json.dumps(summary, indent=2))
    for name, message in state["failed"].items():
        print(f"FAILED {name}: {message}")


if __name__ == "__main__":
    main()
