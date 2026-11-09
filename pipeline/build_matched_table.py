import argparse

from rivwq.matching import readings, table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gee-run-id", required=True)
    parser.add_argument("--sites", nargs="*")
    parser.add_argument("--levels", nargs="*")
    args = parser.parse_args()
    config = readings.load_config()
    levels = args.levels or [config["qc_primary"], *config["qc_sensitivity"]]
    out_dir = table.run(config, args.gee_run_id, levels, args.sites, readings.REPO_ROOT)
    print(out_dir)


if __name__ == "__main__":
    main()
