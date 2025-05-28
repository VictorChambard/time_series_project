import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imports internes
from etl.extract import load_config, download_multiple_stocks
from etl.transform import clean_multivariate_data
from models.forecast import (
    estimer_var,
    estimer_vecm,
    preparer_rendements_log_mensuels,
    choisir_modele,
    tester_stationnarite_et_coint
)
from helpers.viz import (
    plot_adf_test,
    plot_cointegration_test,
    plot_forecast,
    plot_vecm_forecast,
    plot_vol_vs_pred

)
from helpers.volatilite import calcul_volatilite


st.set_page_config(page_title="Dashboard Volatilité VIX & S&P500", layout="wide")

# === TITRE ===
st.title("Dashboard de l'Analyse de la volatilité : VIX & S&P500")
st.markdown("Analyse économétrique et prévisions sur la volatilité des marchés financiers.")

# === ÉTAPE 1 : Chargement des données ===
st.header("1. Chargement et aperçu des données")
config = load_config()
df_raw = download_multiple_stocks(config)
df = clean_multivariate_data(df_raw)
st.dataframe(df.head())

# === ÉTAPE 2 : Visualisation des séries temporelles ===
st.header("2. Visualisation des séries temporelles")
cols = st.multiselect("Colonnes à afficher :", df.columns.tolist(), default=df.columns.tolist())

if "Date" in df.columns:
    df_to_plot = df.set_index("Date")
else:
    df_to_plot = df.copy()

valid_cols = [col for col in cols if col in df_to_plot.columns]

if valid_cols:
    st.line_chart(df_to_plot[valid_cols])
else:
    st.warning("Aucune des colonnes sélectionnées n'est valide.")

# === ÉTAPE 3 : Tests de stationnarité et co-intégration ===
st.header("3. Tests de stationnarité et co-intégration")
if st.button("Lancer les tests de stationnarité et de co-intégration"):
    tester_stationnarite_et_coint()

# === ÉTAPE 4 : Prédiction ===
st.header("4. Prévision de la volatilité")
model_choice = st.selectbox("Choisir le modèle de prévision :", ["Auto (choix basé sur les tests)", "VAR", "VECM"])

if st.button("Lancer la prévision"):
    if model_choice == "Auto (choix basé sur les tests)":
        selected_model = choisir_modele()
        st.write("Modèle sélectionné automatiquement :", selected_model)
    else:
        selected_model = model_choice

    if selected_model == "VAR":
        rendements = preparer_rendements_log_mensuels()
        resultats_var = estimer_var(rendements)
        st.subheader("Prévision avec modèle VAR")
        plot_forecast(resultats_var.fittedvalues, title="Prévision VAR en échantillon")

    elif selected_model == "VECM":
        resultat_vecm = estimer_vecm()
        st.subheader("Prévision avec modèle VECM")
        plot_vecm_forecast(resultat_vecm)

# === ÉTAPE 5 : Conclusion ===
st.header("5. Conclusion")
st.markdown("""
- Les tests ADF et Johansen permettent de diagnostiquer la stationnarité et les relations de long terme.
- Les modèles VAR sont adaptés aux séries stationnaires.
- Le VECM permet d’exploiter les relations de co-intégration pour des séries non stationnaires.
- Le modèle est sélectionné automatiquement selon les résultats des tests.
""")

# === ÉTAPE 6 : Comparaison ===

st.header("6. Comparaison : Volatilité Réelle vs Prévision VIX")
# Extraction des vraies valeurs de VIX et calcul de sa volatilité
vix_real = df["VIX_Close"]
vol_realisee = calcul_volatilite(vix_real)

# Prévision du modèle VAR ou VECM (selon le modèle sélectionné)
if selected_model == "VAR":
    pred_vix = resultats_var.fittedvalues["VIX_Close"]
elif selected_model == "VECM":
    pred_vix = result_vcm.predict(steps=len(df))[:, 0]  # 0 si VIX est en 1ère colonne
    pred_vix = pd.Series(pred_vix, index=df.index[-len(pred_vix):])

# Affichage du graphique comparatif
plot_vol_vs_pred(vol_realisee, pred_vix)
