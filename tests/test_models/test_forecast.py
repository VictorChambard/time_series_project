import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECMResults

from models.forecast import (
    preparer_rendements_log_mensuels,
    estimer_var,
    estimer_vecm,
    choisir_modele,
)

def test_preparer_rendements_log_mensuels():
    """
    Vérifie que la préparation des rendements log mensuels retourne un DataFrame propre
    avec les bonnes colonnes et sans valeurs manquantes.
    """
    df = preparer_rendements_log_mensuels()
    assert isinstance(df, pd.DataFrame), "La fonction ne renvoie pas un DataFrame"
    assert not df.isnull().values.any(), "Le DataFrame contient des valeurs manquantes"
    assert "VIX_Close" in df.columns and "GSPC_Close" in df.columns, "Colonnes manquantes dans les rendements"

def test_estimer_var():
    """
    Vérifie que le modèle VAR est bien estimé et possède les attributs attendus.
    """
    rendements = preparer_rendements_log_mensuels()
    modele = estimer_var(rendements)
    assert hasattr(modele, 'fittedvalues'), "Le modèle VAR ne semble pas avoir été ajusté correctement"

def test_estimer_vecm():
    """
    Vérifie que la fonction retourne bien un objet de type VECMResults.
    """
    vecm = estimer_vecm()
    assert isinstance(vecm, VECMResults), "Le modèle VECM n’est pas du bon type"

def test_choisir_modele():
    """
    Vérifie que la fonction retourne bien une des deux chaînes attendues : 'VAR' ou 'VECM'.
    """
    modele = choisir_modele()
    assert modele in ["VAR", "VECM"], f"Le modèle retourné n’est pas reconnu : {modele}"
