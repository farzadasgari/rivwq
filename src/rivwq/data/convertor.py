import pandas as pd
import numpy as np


def read_inputs(field_csv: str, sentinel_csv: str):
    df_field = pd.read_csv(field_csv)
    df_satel = pd.read_csv(sentinel_csv)

    df_field["Date"] = pd.to_datetime(df_field["Date"])
    df_satel["Date"] = pd.to_datetime(df_satel["Date"])
    return df_field, df_satel


def convert_units(df_field: pd.DataFrame) -> pd.DataFrame:
    df_field = df_field.copy()
    df_field["Discharge"] = df_field["Discharge"] * 0.0283168
    df_field["Height"] = df_field["Height"] * 0.3048
    return df_field


def scale_sentinel(df_satel: pd.DataFrame) -> pd.DataFrame:
    df_satel = df_satel.copy()

    band_cols = [
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12",
        "TCI_B", "TCI_G", "TCI_R"
    ]
    aot_wvp_cols = ["AOT", "WVP"]

    for col in band_cols:
        if col in df_satel.columns:
            df_satel[f"{col}_scaled"] = np.maximum(df_satel[col] / 10000, 0)

    for col in aot_wvp_cols:
        if col in df_satel.columns:
            df_satel[f"{col}_scaled"] = df_satel[col] / 1000

    df_satel = df_satel.drop(
        columns=[col for col in band_cols + aot_wvp_cols if col in df_satel.columns])

    rename_dict = {
        f"{col}_scaled": col
        for col in band_cols + aot_wvp_cols
        if f"{col}_scaled" in df_satel.columns
    }
    df_satel = df_satel.rename(columns=rename_dict)

    return df_satel


def compute_indices(merged_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = merged_df.copy()

    merged_df["MNDWI"] = (merged_df["B3"] - merged_df["B11"]) / \
        (merged_df["B3"] + merged_df["B11"])
    merged_df["GNDVI"] = (merged_df["B8"] - merged_df["B3"]) / \
        (merged_df["B8"] + merged_df["B3"])
    merged_df["SDDI"] = np.log(merged_df["B3"] / merged_df["B4"])
    merged_df["NDTI"] = (merged_df["B4"] - merged_df["B3"]) / \
        (merged_df["B4"] + merged_df["B3"])
    merged_df["BR"] = merged_df["B2"] / merged_df["B4"]
    merged_df["NDWI"] = (merged_df["B3"] - merged_df["B8"]) / \
        (merged_df["B3"] + merged_df["B8"])
    merged_df["NDPI"] = (merged_df["B11"] - merged_df["B3"]) / \
        (merged_df["B11"] + merged_df["B3"])
    merged_df["NDCI"] = (merged_df["B5"] - merged_df["B4"]) / \
        (merged_df["B5"] + merged_df["B4"])
    merged_df["2BDA_Chl"] = merged_df["B5"] - merged_df["B4"]
    merged_df["RR"] = merged_df["B5"] / merged_df["B4"]

    return merged_df


def reorder_columns_like_notebook(df_field: pd.DataFrame, df_satel: pd.DataFrame, merged_df: pd.DataFrame) -> pd.DataFrame:
    field_cols = [col for col in df_field.columns if col != "Date"]
    satel_cols = [col for col in df_satel.columns if col != "Date"]
    index_cols = [
        col for col in merged_df.columns
        if col.startswith(("MNDWI", "GNDVI", "SDDI", "NDTI", "BR", "NDWI", "NDPI", "NDCI", "2BDA_Chl", "RR"))
    ]
    merged_df = merged_df[["Date"] + field_cols + satel_cols + index_cols]
    return merged_df


def make_merged_dataset(field_csv: str, sentinel_csv: str) -> pd.DataFrame:
    df_field, df_satel = read_inputs(field_csv, sentinel_csv)
    df_field = convert_units(df_field)
    df_satel = scale_sentinel(df_satel)

    merged_df = pd.merge(df_field, df_satel, on="Date", how="inner")
    merged_df = compute_indices(merged_df)
    merged_df = reorder_columns_like_notebook(df_field, df_satel, merged_df)
    return merged_df
