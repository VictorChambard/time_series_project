import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

from helpers.viz import plot_adf_test, plot_cointegration_test, plot_forecast , plot_vecm_forecast
from helpers.volatilite import calcul_log_rendements


def charger_donnees():
    chemin_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chemin_fichier = os.path.join(chemin_base, "data", "processed", "clean_data.csv")
    df = pd.read_csv(chemin_fichier)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df


def tester_stationnarite_et_coint():
    donnees = charger_donnees()
    log_prix = np.log(donnees[["VIX_Close", "GSPC_Close"]])

    print("Test de stationnarité (ADF)")
    resultats_adf = {}

    for nom_colonne in log_prix.columns:
        serie = log_prix[nom_colonne]
        test = adfuller(serie)
        p_valeur = test[1]

        resultats_adf[nom_colonne] = {"p-value": p_valeur}

        if p_valeur < 0.05:
            conclusion = "stationnaire"
        else:
            conclusion = "non stationnaire"

        print(nom_colonne, ": p-value =", round(p_valeur, 4), "→", conclusion)

    plot_adf_test(resultats_adf)

    print("\nTest de co-intégration (Johansen)")
    resultat_johansen = coint_johansen(log_prix, det_order=0, k_ar_diff=2)

    statistiques = resultat_johansen.lr1
    valeurs_critiques = resultat_johansen.cvt

    for i in range(len(statistiques)):
        print("H0 : au plus", i, "relation(s) co-intégrée(s)")
        print("Statistique =", round(statistiques[i], 2), "/ seuil 5% =", valeurs_critiques[i, 1])
        if statistiques[i] > valeurs_critiques[i, 1]:
            print("→ H0 rejetée, il y a au moins", i + 1, "relation(s) co-intégrée(s)\n")
        else:
            print("→ H0 non rejetée\n")

    plot_cointegration_test(statistiques, valeurs_critiques)


def preparer_rendements_log_mensuels():
    df = charger_donnees()
    df = df[["VIX_Close", "GSPC_Close"]]
    df_mensuel = df.resample("M").last()

    log_rendements = pd.DataFrame()

    for colonne in df_mensuel.columns:
        serie = df_mensuel[colonne]
        rendements = calcul_log_rendements(serie)
        log_rendements[colonne] = rendements

    log_rendements.dropna(inplace=True)
    return log_rendements


def estimer_var(rendements, max_lags=12):
    modele = VAR(rendements)
    selection = modele.select_order(maxlags=max_lags)
    print(selection.summary())

    meilleur_lag = selection.selected_orders["aic"]
    estimation = modele.fit(meilleur_lag)
    return estimation


def afficher_forecast_var(resultats):
    previsions = resultats.fittedvalues
    plot_forecast(previsions, title="Prévision VAR en échantillon")


def estimer_vecm(lags_diff=2, coint_rank=1):
    df = charger_donnees()
    log_prix = np.log(df[["VIX_Close", "GSPC_Close"]])
    modele = VECM(log_prix, k_ar_diff=lags_diff, coint_rank=coint_rank, deterministic="co")
    resultat = modele.fit()
    print("Résumé du modèle VECM :")
    print(resultat.summary())
    return resultat


if __name__ == "__main__":
    print("Tests de stationnarité et co-intégration")
    tester_stationnarite_et_coint()

    MODELE = "VECM"

    if MODELE == "VAR":
        print("Préparation des log-rendements mensuels")
        rendements = preparer_rendements_log_mensuels()

        print("Estimation du modèle VAR")
        resultat_var = estimer_var(rendements)

        print("Prévision à partir du modèle VAR")
        afficher_forecast_var(resultat_var)

    elif MODELE == "VECM":
        print("Estimation du modèle VECM")
        resultat_vecm = estimer_vecm()

        print("Prévision à partir du modèle VECM")
        plot_vecm_forecast(resultat_vecm)

def choisir_modele(): 
    """
    Analyse des tests de stationnarité et de co-intégration pour choisir le modèle approprié.

    Args:
        Aucune 
    Returns : 
        str : Le nom du modèle choisi ("VAR" ou "VECM").
    """
    donnees = charger_donnees()
    log_prix = np.log(donnees[["VIX_Close", "GSPC_Close"]])

    series_non_stationnaires = []
    for colonne in log_prix.columns:
        serie = log_prix[colonne]
        resultat = adfuller(serie)
        p_valeur = resultat[1]
        if p_valeur > 0.05:
            series_non_stationnaires.append(colonne)

    # Test de co-intégration de Johansen
    test_johansen = coint_johansen(log_prix, det_order=0, k_ar_diff=2)
    stats_trace = test_johansen.lr1
    valeurs_critiques = test_johansen.cvt

    cointegration_detectee = False
    for i in range(len(stats_trace)):
        seuil_5 = valeurs_critiques[i, 1]
        if stats_trace[i] > seuil_5:
            cointegration_detectee = True

    # Logique de sélection du modèle
    if len(series_non_stationnaires) > 0 and cointegration_detectee:
        return "VECM"
    else:
        return "VAR"