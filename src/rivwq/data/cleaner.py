import pandas as pd


def detect_outliers_iqr(series: pd.Series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers, lower_bound, upper_bound


def load_merged_dataset(merged_csv: str) -> pd.DataFrame:
    df = pd.read_csv(merged_csv)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def add_band_out_of_bounds_flags(df: pd.DataFrame, band_cols=None) -> pd.DataFrame:
    df = df.copy()
    if band_cols is None:
        band_cols = ["B1", "B2", "B3", "B4", "B5",
                     "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

    for col in band_cols:
        if col in df.columns:
            df[f"{col}_outlier_bound"] = (df[col] < 0) | (df[col] > 1)

    return df


def add_iqr_outlier_flags(df: pd.DataFrame, priority_cols) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    outlier_summary = []

    for col in priority_cols:
        if col in df.columns:
            outliers, lb, ub = detect_outliers_iqr(df[col].dropna())
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(df)) * 100

            outlier_summary.append(
                {
                    "Column": col,
                    "Outlier_Count": outlier_count,
                    "Outlier_Percentage": round(outlier_pct, 2),
                    "Lower_Bound": round(lb, 4),
                    "Upper_Bound": round(ub, 4),
                }
            )

            df[f"{col}_outlier"] = (df[col] < lb) | (df[col] > ub)

    summary_df = pd.DataFrame(outlier_summary)
    return df, summary_df


def clean_dataset(df: pd.DataFrame, priority_cols) -> tuple[pd.DataFrame, pd.DataFrame]:
    band_cols = ["B1", "B2", "B3", "B4", "B5", "B6",
                 "B7", "B8", "B8A", "B9", "B11", "B12"]

    df = add_band_out_of_bounds_flags(df, band_cols=band_cols)
    df, summary_df = add_iqr_outlier_flags(df, priority_cols=priority_cols)

    outlier_flags = [
        f"{col}_outlier" for col in priority_cols if f"{col}_outlier" in df.columns]
    outlier_flags += [
        f"{col}_outlier_bound" for col in band_cols if f"{col}_outlier_bound" in df.columns]

    df["has_outlier"] = df[outlier_flags].any(axis=1)

    cleaned_df = df[~df["has_outlier"]].copy()

    cleaned_df = cleaned_df.drop(
        columns=[
            col
            for col in df.columns
            if col.endswith("_outlier") or col.endswith("_outlier_bound") or col == "has_outlier"
        ]
    )

    return cleaned_df, summary_df


def run_cleaning(merged_csv: str, priority_cols) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_merged_dataset(merged_csv)
    cleaned_df, summary_df = clean_dataset(df, priority_cols=priority_cols)
    return df, cleaned_df, summary_df
