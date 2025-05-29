import matplotlib.pyplot as plt
import streamlit as st

def plot_irf_var(resultats_var, horizon=10):
    """
    Trace les fonctions de réponse impulsionnelle (IRF) du modèle VAR estimé.

    Args:
        resultats_var : objet VARResults
        horizon (int) : nombre de périodes pour l'IRF
    """
    irf = resultats_var.irf(horizon)
    fig = irf.plot(orth=True)
    fig.suptitle("Réponses impulsionnelles (IRF) - VAR")
    plt.tight_layout()
    st.pyplot(fig)
