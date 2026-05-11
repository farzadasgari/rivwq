import argparse
from pathlib import Path

from sedis.data.convertor import make_merged_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare merged dataset (convert units, scale sentinel, compute indices).")
    p.add_argument("--field", default="data/raw/dataset/field.csv")
    p.add_argument("--sentinel", default="data/raw/dataset/sentinel.csv")
    p.add_argument("--out", default="data/processed/dataset/merged.csv")
    return p.parse_args()


def main():
    args = parse_args()

    df = make_merged_dataset(args.field, args.sentinel)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("Merged data with unit conversions and indices exported.")
    print(f"Saved -> {out.resolve()}")


if __name__ == "__main__":
    main()
