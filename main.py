from etl.extract import load_config, download_multiple_stocks
from etl.transform import clean_multivariate_data
from etl.load import save_to_csv
import pandas as pd

def run_etl() -> pd.DataFrame:
    config = load_config()
    raw_df = download_multiple_stocks(config)
    clean_df = clean_multivariate_data(raw_df)
    output_path = config["data_paths"]["processed"] + "clean_data.csv"
    save_to_csv(clean_df, output_path)
    return clean_df

if __name__ == "__main__":
    print("🚀 Lancement du pipeline ETL...")
    cleaned_df = run_etl()
    print("✅ ETL terminé avec succès.")
    print(cleaned_df.head())