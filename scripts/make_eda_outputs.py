import argparse
from pathlib import Path

from rivwq.eda.io import load_clean_data
from rivwq.eda.stats import descriptive_table, shapiro_table, kendall_tau_matrix
from rivwq.eda.plots import set_paper_style, corr_heatmap, distplots, qqplots


def parse_args():
    p = argparse.ArgumentParser(description="Generate EDA tables + plots from dataset.")
    p.add_argument("--in_csv", default="data/processed/dataset/data.csv")
    p.add_argument("--results_dir", default="results/eda")
    p.add_argument("--figures_dir", default="figures/eda")
    p.add_argument("--annot", action="store_true", help="Annotate heatmap values (slower / busy).")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_clean_data(args.in_csv)

    core = ["Discharge", "Height", "DO", "fChl", "Temperature", "Turbidity"]

    spectral = [
        "B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12",
        "TCI_B","TCI_G","TCI_R","AOT","WVP",
        "MNDWI","GNDVI","SDDI","NDTI","BR","NDWI","NDPI","NDCI","2BDA_Chl","RR",
    ]

    core = [c for c in core if c in df.columns]
    spectral = [c for c in spectral if c in df.columns]

    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir); figures_dir.mkdir(parents=True, exist_ok=True)

    desc = descriptive_table(df, cols=core + spectral)
    desc.to_csv(results_dir / "descriptive_all.csv")

    shapiro_core = shapiro_table(df, cols=core)
    shapiro_core.to_csv(results_dir / "shapiro_core.csv", index=False)

    corr_cols = core + spectral
    corr = kendall_tau_matrix(df, cols=corr_cols)
    corr.to_csv(results_dir / "kendall_tau_matrix.csv")


    set_paper_style()

    corr_heatmap(
        corr,
        out_path=figures_dir / "kendall_tau_heatmap_triangular.png",
        title="Kendall τ (lower triangle)",
        annot=args.annot,
        mask_upper=True,
    )

    distplots(df, core, figures_dir / "dist_core.png", bins=30, kde=True)

    chunk_size = 12
    for i in range(0, len(spectral), chunk_size):
        chunk = spectral[i:i+chunk_size]
        distplots(df, chunk, figures_dir / f"dist_spectral_{i//chunk_size+1:02d}.png", bins=30, kde=True)

    qqplots(df, core, figures_dir / "qq_core")

    print("EDA outputs generated:")
    print(f"- Tables: {results_dir.resolve()}")
    print(f"- Figures: {figures_dir.resolve()}")


if __name__ == "__main__":
    main()