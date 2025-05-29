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
from helpers.viz_irf import plot_irf_var
# === CONFIGURATION DE LA PAGE ===


st.set_page_config(page_title="Dashboard Volatilité VIX & S&P500", layout="wide")


# === TITRE ===
st.title("Dashboard de l'Analyse de la volatilité : VIX & S&P500")
st.markdown("""
Ce dashboard explore le lien entre la volatilité implicite du marché (VIX) et l’évolution du S&P500.  
L’objectif est d’analyser statistiquement les dynamiques conjointes à travers des modèles VAR/VECM.

- Le **VIX** est souvent appelé l’« indice de la peur », il reflète l’incertitude des investisseurs.
- Le **S&P500** est un indice représentatif de l’économie américaine.
- Leur relation est souvent utilisée pour **anticiper les périodes de crise**.

Nous cherchons à :
1. Diagnostiquer les propriétés statistiques de ces séries.
2. Choisir un modèle adapté (stationnarité / co-intégration).
3. Comparer les prévisions à la volatilité réalisée (rolling std).
4. Évaluer la qualité de prédiction en période de crise (COVID).
""")
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
st.markdown("En fonction des tests statistiques, le modèle VAR ou VECM est automatiquement sélectionné.")
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

# === ÉTAPE 5 : Conclusion Rapide ===
st.header("5. Conclusion")
st.markdown("""
- Les tests ADF et Johansen permettent de diagnostiquer la stationnarité et les relations de long terme.
- Les modèles VAR sont adaptés aux séries stationnaires.
- Le VECM permet d’exploiter les relations de co-intégration pour des séries non stationnaires.
- Le modèle est sélectionné automatiquement selon les résultats des tests.
""")

# === ÉTAPE 6 : Comparaison avec la volatilité réalisée ===

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

# === ÉTAPE 7 : Crise Covid ===

st.header("7. Focus sur la période COVID (2020-2021)")
st.markdown("""
La crise sanitaire de 2020 a provoqué une volatilité extrême.  
Nous analysons ici comment le modèle se comporte sur cette période critique.
""")
df_zoom = df[["VIX_Close", "GSPC_Close"]].loc["2020":"2021"]
st.line_chart(df_zoom)

# === ÉTAPE 8 : Fonctions de réponse impulsionnelle ===

if selected_model == "VAR":
    st.header("8. Fonctions de réponse impulsionnelle (IRF)")
    st.markdown("""
Les IRF permettent d’analyser l’impact dynamique d’un choc sur une variable (ex : S&P500) sur une autre (ex : VIX).
""")
    plot_irf_var(resultats_var)
elif selected_model == "VECM":
    st.header("8. Fonctions de réponse impulsionnelle (IRF)")
    st.markdown("""
L'analyse des IRF dans un modèle VECM nécessite des outils plus avancés et n’est pas intégrée ici.  
Il serait pertinent de l'ajouter dans une future extension avec orthogonalisation.
""")


# === ÉTAPE 9 : Lecture critique ===

st.header("9. Lecture critique des résultats")
st.markdown("""
- Le VIX capte bien la nervosité des marchés, mais sa prédiction reste difficile en période de choc.
- Les modèles VAR supposent la stationnarité, ce qui peut être une hypothèse forte.
- Les VECM intègrent une relation de long terme mais ne capturent pas les ruptures brutales (comme COVID).
- Une amélioration future pourrait passer par l’introduction de ruptures structurelles ou de modèles GARCH.
""")

# === ÉTAPE 10 : Informations sur le projet ===
st.header("10. Informations sur le projet")
st.markdown("""
Projet réalisé par le groupe N dans le cadre du cours de Python de Fabrice Galan, avec Streamlit et Python.

**Membres du groupe :**  
- Victor Chambard  
- Niama El Kamal  
- Youssef Saied

**Données :** Yahoo Finance · Période : 2018–2024.  
**Référence académique utilisée :**  
- Zhang, B., Hu, Y., & Ji, Q. (2020). Financial markets under the global pandemic of COVID-19. *Finance Research Letters*, 36, 101528.
- Becker, R., Clements, A. E., & McClelland, A. (2022). *Forecasting Realized Volatility and VIX: A comparison of VAR and Machine Learning Models*.
""")
