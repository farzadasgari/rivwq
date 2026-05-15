from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import shapiro, kendalltau


def numeric_columns(df: pd.DataFrame, exclude=("Date",)) -> list[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def descriptive_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]


def shapiro_table(df: pd.DataFrame, cols: list[str], max_n: int = 5000) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        x = df[c].dropna().to_numpy()
        if len(x) == 0:
            continue
        if len(x) > max_n:
            x = x[:max_n]
        W, p = shapiro(x)
        rows.append({"column": c, "n_used": len(
            x), "W": float(W), "p_value": float(p)})
    return pd.DataFrame(rows).sort_values("p_value")


def kendall_tau_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    m = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j < i:
                continue
            x = df[a]
            y = df[b]
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                tau = np.nan
            else:
                tau, _ = kendalltau(x[valid], y[valid])
            m.loc[a, b] = tau
            m.loc[b, a] = tau
    return m


def pearson_corr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].corr(method="pearson")
