import pandas as pd
import os

def save_to_csv(df: pd.DataFrame, path: str) -> None:
    """
Enregistre un DataFrame Pandas dans un fichier CSV.
    Args:
        df (pd.DataFrame): Le DataFrame à enregistrer.
        path (str): Chemin du fichier CSV de destination.
    Returns: 
        Ne retourne rien.
    """
    # Crée le dossier qui n'existe pas
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Sauvegarde en CSV
    df.to_csv(path, index=False)
    print(f"Donnée Sauvegardées dans : {path}")

# Pour test
if __name__ == "__main__":
    from extract import load_config, download_stock_data
    from transform import clean_stock_data

    config = load_config()
    raw_df = download_stock_data(config)
    clean_df = clean_stock_data(raw_df)

    output_path = config["data_paths"]["processed"] + "clean_data.csv"
    save_to_csv(clean_df, output_path)