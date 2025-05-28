from etl.extract import load_config, download_multiple_stocks
from etl.transform import clean_multivariate_data
from etl.load import save_to_csv
from models.forecast import (
    preparer_rendements_log_mensuels,
    estimer_var,
    afficher_forecast_var,
    estimer_vecm,
    choisir_modele
)
import pandas as pd

def run_etl() -> pd.DataFrame:
    config = load_config()
    raw_df = download_multiple_stocks(config)
    clean_df = clean_multivariate_data(raw_df)
    output_path = config["data_paths"]["processed"] + "clean_data.csv"
    save_to_csv(clean_df, output_path)
    return clean_df

if __name__ == "__main__":

    print(" Lancement du pipeline ETL")
    cleaned_df = run_etl()
    print(" ETL terminé avec succès.")
    print(cleaned_df.head())
    print("Analyse des propriétés des séries pour choisir le modèle")
    modele = choisir_modele()
    print("Modèle choisi :", modele)

    if modele == "VAR":
        print("Préparation des rendements log mensuels")
        rendements = preparer_rendements_log_mensuels()
        print("Estimation du modèle VAR")
        resultat = estimer_var(rendements)
        print("Affichage des prévisions VAR")
        afficher_forecast_var(resultat)

    elif modele == "VECM":
        print("Estimation du modèle VECM")
        resultat_vecm = estimer_vecm()
    
    import os

    # Lancement du dashboard Streamlit
    os.system("streamlit run app/dashboard.py")
