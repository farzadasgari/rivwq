import argparse
from pathlib import Path

from rivwq.data.cleaner import run_cleaning


def parse_args():
    p = argparse.ArgumentParser(description="Clean merged dataset using IQR outliers + band [0,1] bounds.")
    p.add_argument("--in_csv", default="data/processed/dataset/merged.csv")
    p.add_argument("--out_csv", default="data/processed/dataset/data_cleaned.csv")
    p.add_argument("--summary_csv", default="results/outlier_summary.csv")
    return p.parse_args()


def main():
    args = parse_args()

    priority_cols = [
        "Discharge", "Height", "Turbidity", "pH", "DO", "SC", "Temperature",
        "fChl", "Phycocyanin",
        "B2", "B3", "B4", "B5", "B8", "B11",
    ]

    df, cleaned_df, summary_df = run_cleaning(args.in_csv, priority_cols=priority_cols)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(out_csv, index=False)

    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nOriginal rows: {len(df)}")
    print(f"Cleaned rows: {len(cleaned_df)}")
    removed = len(df) - len(cleaned_df)
    print(f"Rows removed: {removed} ({(removed / len(df) * 100):.2f}%)")
    print(f"Cleaned data exported -> {out_csv.resolve()}")
    print(f"Outlier summary exported -> {summary_csv.resolve()}")


if __name__ == "__main__":
    main()