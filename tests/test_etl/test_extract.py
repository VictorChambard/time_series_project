import pandas as pd
from etl.extract import load_config, download_multiple_stocks

def test_telechargement_donnees():
    """
    Vérifie que la fonction download_multiple_stocks retourne un DataFrame non vide.
    """
    config = load_config()
    donnees_boursieres = download_multiple_stocks(config)

    # Vérification du type
    assert isinstance(donnees_boursieres, pd.DataFrame), "La fonction ne retourne pas un DataFrame"

    # Vérification du contenu
    assert not donnees_boursieres.empty, "Le DataFrame retourné est vide"

    # Vérification des colonnes attendues
    colonnes_attendues = {"VIX_Close", "GSPC_Close"}
    assert colonnes_attendues.issubset(set(donnees_boursieres.columns)), "Colonnes manquantes dans les données"