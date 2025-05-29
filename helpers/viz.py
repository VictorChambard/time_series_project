import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def plot_adf_test(series_dict):
    """ 
    Affiche les p-valeurs du test ADF pour chaque série temporelle.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, result in series_dict.items():
        ax.bar(name, result['p-value'], color="green" if result['p-value'] < 0.05 else "red")
    ax.axhline(y=0.05, linestyle='--', color='black', label="Seuil 5%")
    ax.set_title("P-valeurs du test ADF")
    ax.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

def plot_cointegration_test(trace_stats, crit_values):
    """ 
    Affiche les statistiques de trace Johansen vs. seuils critiques.
    """
    x = range(len(trace_stats))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, trace_stats, color='blue', alpha=0.7)

    for i in x:
        ax.hlines(crit_values[i, 1], xmin=i - 0.4, xmax=i + 0.4, colors='red', linestyles='--')

    ax.set_title("Test de Johansen - Statistiques Trace vs Critique 5%")
    ax.set_xlabel("Nombre de relations co-intégrées")
    ax.set_ylabel("Statistique")
    ax.legend(["Valeur critique (5%)"])
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

def plot_forecast(df, title="Prévision des log-rendements"):
    """ 
    Affiche les prévisions d’un modèle sur plusieurs variables.
    """
    plt.figure(figsize=(10, 5))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Valeur prévue")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

def plot_vecm_forecast(vecm_resultat, pas=12):
    """ 
    Affiche les prévisions du modèle VECM sur une période donnée.
    """
    previsions = vecm_resultat.predict(steps=pas)

    index_futur = pd.date_range(
        start=vecm_resultat.model.data.row_labels[-1],
        periods=pas + 1,
        freq="ME"
    )[1:]

    df_previsions = pd.DataFrame(previsions, columns=vecm_resultat.names, index=index_futur)

    plt.figure(figsize=(10, 5))
    for colonne in df_previsions.columns:
        plt.plot(df_previsions.index, df_previsions[colonne], label=colonne)

    plt.title("Prévision VECM sur " + str(pas) + " mois")
    plt.xlabel("Date")
    plt.ylabel("Valeurs prévues")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

def plot_vol_vs_pred(real_vol, pred):
    import matplotlib.pyplot as plt
    import pandas as pd
    df_plot = pd.concat([real_vol, pred], axis=1).dropna()
    real_vol_clean = df_plot.iloc[:, 0]
    pred_clean = df_plot.iloc[:, 1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(real_vol_clean.index, real_vol_clean, label="Volatilité Réelle")
    ax.plot(pred_clean.index, pred_clean, label="Prévision VIX")
    ax.set_title("Volatilité Réelle vs Prévision VIX")
    ax.legend()
    st.pyplot(fig)

def plot_irf_var(resultats_var, pas=12):
    """
    Affiche les fonctions de réponse impulsionnelle (IRF) pour un modèle VAR.

    Args:
        resultats_var : Résultat de l’estimation VAR (objet VARResults).
        pas (int) : Nombre de périodes à simuler.
    """
    irf = resultats_var.irf(pas)  # On génère les IRF sur "pas" périodes
    fig = irf.plot(orth=False)    # Affichage simple des IRF sans orthogonalisation
    fig.tight_layout()            # Pour éviter les chevauchements
    plt.show()                    # Affichage des graphes
