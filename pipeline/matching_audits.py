import argparse
from pathlib import Path

from rivwq.matching import audits, readings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="*")
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    config = readings.load_config()
    result = audits.run(config, readings.REPO_ROOT, args.sites)
    report_dir = args.report_dir or Path(config["report_dir"]).expanduser()
    for path in audits.write(result, report_dir):
        print(path)


if __name__ == "__main__":
    main()
