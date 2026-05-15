import pandas as pd


def load_clean_data(csv_path: str, date_column: str = "Date") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
    return df
