import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calcul_log_rendements(prix):
    """
    Calcule les rendements logarithmiques à partir d'une série de prix.

    Args:
        prix (pd.Series): Série de prix (exemple : S&P500 ou VIX).

    Returns:
        pd.Series: Série des rendements logarithmiques, alignée avec les dates.
    """
    rendements = []

    for i in range(1, len(prix)):
        if prix[i - 1] != 0:
            variation = (prix[i] - prix[i - 1]) / prix[i - 1]
            log_r = np.log(1 + variation)
            rendements.append(log_r)
        else:
            rendements.append(0)

    rendements = [pd.NA] + rendements  
    return pd.Series(rendements, index=prix.index)


def calcul_volatilite(prix, fenetre=21):
    """
    Calcule la volatilité réalisée annualisée à partir d'une série de prix.
    Utilise les rendements logarithmiques, puis applique un écart-type glissant.

    Args:
        prix (pd.Series): Série de prix (exemple : cours de clôture).
        fenetre (int): Taille de la fenêtre mobile en jours (par défaut 21).

    Returns:
        pd.Series: Série de volatilité réalisée annualisée.
    """
    rendements_log = calcul_log_rendements(prix)
    volatilite = []

    for i in range(len(rendements_log)):
        if i < fenetre:
            volatilite.append(pd.NA)
        else:
            donnees = rendements_log[i - fenetre:i]
            ecart_type = donnees.std()
            volatilite_annualisee = ecart_type * (252 ** 0.5)
            volatilite.append(volatilite_annualisee)

    return pd.Series(volatilite, index=prix.index, name="Volatilite_Annualisee")
