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

    # Séparation des colonnes pour simuler deux sources
    vix = raw_df["VIX_Close"]
    gspc = raw_df["GSPC_Close"]


    # Jointure explicite
    merged_df = join_datasets(vix, gspc)

    # Nettoyage
    clean_df = clean_multivariate_data(merged_df)

    print(" Aperçu des données nettoyées et jointes :")
    print("Aperçu des données nettoyées et jointes :")
    print(clean_df.head())

