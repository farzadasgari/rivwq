from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from scipy import stats


def set_style():
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 16,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 16,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


def _savefig(fig: plt.Figure, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _no_grid(ax):
    ax.grid(False)


def corr_heatmap(
    corr: pd.DataFrame,
    out_path: str | Path,
    *,
    cmap: str = "coolwarm",
    mask_upper: bool = True,
    vmin: float = -1,
    vmax: float = 1,
    annot: bool = False,
    fmt: str = ".2f",
):
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else np.tril(
        np.ones_like(corr, dtype=bool))
    np.fill_diagonal(mask, False)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        linewidths=0.5,
        cbar=True,
        annot=annot,
        fmt=fmt,
        ax=ax,
    )
    _no_grid(ax)
    _savefig(fig, out_path)


def distplots(
    df: pd.DataFrame,
    cols: list[str],
    out_path: str | Path,
    *,
    bins: int = 30,
    kde: bool = True,
    ncols: int = 3,
):
    cols = [c for c in cols if c in df.columns]
    n = len(cols)
    if n == 0:
        raise ValueError("No columns provided exist in df.")

    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, c in enumerate(cols):
        ax = axes[i]
        sns.histplot(df[c].dropna(), bins=bins, kde=kde, ax=ax)
        _no_grid(ax)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    _savefig(fig, out_path)


def qqplots(
    df: pd.DataFrame,
    columns: list[str],
    out_dir: str | Path,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in columns:
        if col not in df.columns:
            continue

        x = df[col].dropna()
        if len(x) < 3:
            continue

        plt.figure(figsize=(10, 4))

        ax1 = plt.subplot(1, 2, 1)
        sns.kdeplot(x, color="navy", linewidth=1.5, ax=ax1)
        ax1.set_xlabel(col.replace("_", " ").title())
        ax1.set_ylabel("Frequency")
        ax1.grid(False)

        ax2 = plt.subplot(1, 2, 2)
        qqplot(
            x,
            line="s",
            dist=stats.norm,
            ax=ax2,
            marker=".",
            markersize=12,
            alpha=0.6,
        )
        ax2.set_xlabel("Theoretical Quantiles")
        ax2.set_ylabel("Sample Quantiles")
        ax2.grid(False)

        plt.tight_layout()

        plt.savefig(
            out_dir / f"{col}.png",
            format="png",
            bbox_inches="tight",
            dpi=600,
        )
        plt.close()


def pairplot_kde_diag(
    df: pd.DataFrame,
    cols: list[str],
    out_path: str | Path,
    *,
    diag_kind: str = "kde",
    lower_kind: str = "scatter",
    sample: int | None = 2000,
    alpha: float = 0.5,
    s: float = 40,
    kde_fill: bool = False,
):

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        raise ValueError("Need at least 2 columns for pairplot.")

    data = df[cols].dropna()
    if sample is not None and len(data) > sample:
        data = data.sample(n=sample, random_state=0)

    g = sns.PairGrid(data, vars=cols, corner=False, diag_sharey=False)

    if diag_kind == "kde":
        g.map_diag(sns.kdeplot, color="navy", linewidth=2, fill=kde_fill)
    else:
        raise ValueError("Only diag_kind='kde' is supported.")

    if lower_kind == "scatter":
        g.map_lower(sns.scatterplot, s=s, alpha=alpha,
                    color="navy", edgecolor=None)
    elif lower_kind == "kde":
        g.map_lower(sns.kdeplot, levels=5, color="navy", linewidths=1)
    else:
        raise ValueError("lower_kind must be 'scatter' or 'kde'.")

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax = g.axes[i, j]
            if ax is None:
                continue
            ax.grid(False)
            ax.set_title("")

            if j > i:
                ax.set_visible(False)

    g.fig.set_size_inches(2.6 * len(cols), 2.6 * len(cols))
    plt.tight_layout()
    g.fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(g.fig)
