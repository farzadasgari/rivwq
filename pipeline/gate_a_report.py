import argparse
from pathlib import Path

from rivwq.matching import gate_a, readings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gee-run-id", required=True)
    parser.add_argument("--level")
    parser.add_argument("--sites", nargs="*")
    parser.add_argument("--prefix", default="06_")
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    config = readings.load_config()
    level = args.level or config["qc_primary"]
    result = gate_a.run(config, args.gee_run_id, level, args.sites, readings.REPO_ROOT)
    report_dir = args.report_dir or Path(config["report_dir"]).expanduser()
    for path in gate_a.write(result, report_dir, args.prefix):
        print(path)


if __name__ == "__main__":
    main()
