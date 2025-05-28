import pandas as pd
from etl.extract import load_config, download_multiple_stocks
from etl.transform import clean_multivariate_data

def test_nettoyage_donnees():
    """
    Vérifie que la fonction clean_multivariate_data retourne un DataFrame propre.
    """
    config = load_config()
    donnees_boursieres = download_multiple_stocks(config)

    donnees_nettoyees = clean_multivariate_data(donnees_boursieres)

    # Vérification du type
    assert isinstance(donnees_nettoyees, pd.DataFrame), "La fonction ne retourne pas un DataFrame"

    # Vérifie l'absence de valeurs manquantes
    assert not donnees_nettoyees.isnull().values.any(), "Il reste des valeurs manquantes dans le DataFrame"

    # Vérifie que la colonne Date est bien de type datetime
    assert pd.api.types.is_datetime64_any_dtype(donnees_nettoyees["Date"]), "La colonne 'Date' n'est pas de type datetime"