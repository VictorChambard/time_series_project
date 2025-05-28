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

def join_datasets(vix: pd.Series, gspc: pd.Series) -> pd.DataFrame:
    """
    Effectue une jointure explicite entre les séries VIX et GSPC sur la colonne Date.
    """
    vix_df = vix.reset_index()[["Date", "VIX_Close"]]
    gspc_df = gspc.reset_index()[["Date", "GSPC_Close"]]
    return pd.merge(vix_df, gspc_df, on="Date", how="inner")

if __name__ == "__main__":
    from extract import load_config, download_multiple_stocks

    config = load_config()
    raw_df = download_multiple_stocks(config)

    # Séparation des colonnes pour simuler deux sources
    vix = raw_df["VIX_Close"]
    gspc = raw_df["GSPC_Close"]
    vix = df_raw["VIX_Close"]
    gspc = df_raw["GSPC_Close"]

    # Jointure explicite
    merged_df = join_datasets(vix, gspc)

    # Nettoyage
    clean_df = clean_multivariate_data(merged_df)

    print(" Aperçu des données nettoyées et jointes :")
    print("✅ Aperçu des données nettoyées et jointes :")
    print(clean_df.head())

