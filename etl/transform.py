import pandas as pd

def clean_multivariate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie un DataFrame contenant plusieurs séries financières.
    - Convertit Date en datetime
    - Supprime les valeurs manquantes
    - Trie les dates
    """
    df["Date"] = pd.to_datetime(df["Date"])
    df.dropna(inplace=True)
    df.sort_values("Date", inplace=True)
    return df

if __name__ == "__main__":
    from extract import load_config, download_multiple_stocks

    config = load_config()
    raw_df = download_multiple_stocks(config)
    clean_df = clean_multivariate_data(raw_df)

    print(clean_df.head())

