import matplotlib.pyplot as plt

def plot_adf_test(series_dict):
    """ 
    Affiche les p-valeurs du test ADF pour chaque série temporelle.

    Args:
        series_dict (dict): Dictionnaire contenant les séries temporelles et 
        leurs p-valeurs du test ADF.
     Returns:
        La figure matplotlib affichant les p-valeurs.
        """
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, result in series_dict.items():
        ax.bar(name, result['p-value'], color="green" if result['p-value'] < 0.05 else "red")
    ax.axhline(y=0.05, linestyle='--', color='black', label="Seuil 5%")
    ax.set_title("P-valeurs du test ADF")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_cointegration_test(trace_stats, crit_values):

    x = range(len(trace_stats))
    fig, ax = plt.subplots(figsize=(8, 4))

    # Barres des statistiques
    ax.bar(x, trace_stats, color='blue', alpha=0.7)

    # Lignes horizontales pour chaque valeur critique
    for i in x:
        ax.hlines(crit_values[i, 1], xmin=i - 0.4, xmax=i + 0.4, colors='red', linestyles='--')

    ax.set_title("Test de Johansen - Statistiques Trace vs Critique 5%")
    ax.set_xlabel("Nombre de relations co-intégrées")
    ax.set_ylabel("Statistique")
    ax.legend(["Valeur critique (5%)"])
    plt.tight_layout()
    plt.show()


def plot_forecast(df, title="Prévision des log-rendements"):
    """
    Affiche les prévisions sur plusieurs variables.

    Args:
        df (pd.DataFrame): Résultats de prévision (DataFrame avec les colonnes à afficher)
        title (str): Titre du graphique.
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
    plt.show()

def plot_vecm_forecast(vecm_resultat, pas=12):
    """
    Affiche les prévisions du modèle VECM sur une période donnée.

    Args:
        vecm_resultat (VECMResults): Résultat estimé du modèle VECM.
        pas (int): Nombre de périodes à prévoir (par défaut 12).

    Returns:
        None
    """
    previsions = vecm_resultat.predict(steps=pas)

    index_futur = pd.date_range(
        start=vecm_resultat.model.data.row_labels[-1],
        periods=pas + 1,
        freq="M"
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
    plt.show()
