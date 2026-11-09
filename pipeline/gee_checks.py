import argparse
from pathlib import Path

from rivwq.gee import checks, export, roi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = export.load_config()
    run_dir = roi.REPO_ROOT / config["raw_root"] / args.run_id
    long = checks.load_run(run_dir)
    settings = config["checks"]
    sites = sorted(long["site_no"].unique())
    years = sorted(long["time_utc"].dt.year.unique())
    args.out.mkdir(parents=True, exist_ok=True)
    checks.scene_counts(long).to_csv(args.out / "scene_counts.csv", index=False)
    checks.valid_pixel_loss(long).to_csv(args.out / "valid_pixel_loss.csv", index=False)
    zero = checks.zero_valid_site_years(long, sites, years, settings)
    zero.to_csv(args.out / "zero_valid_site_years.csv", index=False)
    jump = checks.sensor_jump(long, settings, settings["stable_bands"])
    jump.to_csv(args.out / "sensor_jump.csv", index=False)
    matched = checks.matched_comparisons(long, settings, settings["stable_bands"])
    matched.to_csv(args.out / "sensor_matched.csv", index=False)
    print(
        f"rows: {len(long)}, zero-valid site-years: {len(zero)}, flags: {int(jump['flag'].sum())}"
    )


if __name__ == "__main__":
    main()
