import matplotlib.pyplot as plt

def plot_adf_test(series_dict):
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
