import pandas as pd
import numpy as np
import os
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from helpers.viz import plot_adf_test, plot_cointegration_test


def test_stationarity_and_cointegration():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "processed", "clean_data.csv")
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    log_prices = np.log(df[["VIX_Close", "GSPC_Close"]])

    print("TEST ADF (stationnarité des log-prix)")
    adf_results = {}
    for col in log_prices.columns:
        result = adfuller(log_prices[col])
        pval = result[1]
        adf_results[col] = {"p-value": pval}
        print(f"{col}: p-value = {pval:.4f} → {'stationnaire' if pval < 0.05 else 'non stationnaire'}")

    # 🟩 Affichage graphique des p-valeurs ADF
    plot_adf_test(adf_results)

    print("\n TEST DE JOHANSEN (co-intégration)")
    johansen_result = coint_johansen(log_prices, det_order=0, k_ar_diff=2)

    trace_stat = johansen_result.lr1
    crit_values = johansen_result.cvt

    for i in range(len(trace_stat)):
        print(f"H0: au plus {i} relation(s) de co-intégration")
        print(f"Statistique trace = {trace_stat[i]:.2f}, Valeur critique (5%) = {crit_values[i, 1]}")
        if trace_stat[i] > crit_values[i, 1]:
            print("→ Rejet de H0 → Il y a au moins", i+1, "relation(s) co-intégrée(s)\n")
        else:
            print("→ On ne rejette pas H0\n")

    # 🟦 Affichage graphique des stats Johansen
    plot_cointegration_test(trace_stat, crit_values)


def prepare_log_returns(data_path: str = "data/processed/clean_data.csv") -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # On garde les colonnes de prix
    df = df[["VIX_Close", "GSPC_Close"]]

    # Log des prix
    log_prices = np.log(df)

    # Resample en mensuel (dernière valeur de chaque mois)
    log_monthly = log_prices.resample("M").last()

    # Log-rendements mensuels
    log_returns = log_monthly.diff().dropna()

    return log_returns

def estimate_var_model(log_returns: pd.DataFrame, lags: int = 1):
    model = VAR(log_returns)
    lag_order = model.select_order(maxlags=12)
    print(lag_order.summary())
    optimal_lag = lag_order.selected_orders["aic"]
    results = model.fit(optimal_lag)
    return results

def plot_in_sample_forecast(results, steps=12):
    forecast = results.fittedvalues

    forecast.plot(figsize=(10, 4))
    plt.title("Prévision en échantillon (in-sample forecast)")
    plt.xlabel("Date")
    plt.ylabel("Log-rendements")
    plt.tight_layout()
    plt.show()


from statsmodels.tsa.vector_ar.vecm import VECM

def estimate_vecm_model(data_path="data/processed/clean_data.csv", lags_diff=2, coint_rank=1):
    # Charger les données
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # Log-prix (attention : pas de différenciation ici)
    log_prices = np.log(df[["VIX_Close", "GSPC_Close"]])

    # Estimation du VECM
    vecm = VECM(log_prices, k_ar_diff=lags_diff, coint_rank=coint_rank, deterministic="co")  # "co" = constante dans le cointegration term
    vecm_res = vecm.fit()

    print("\n Résumé du modèle VECM :\n")
    print(vecm_res.summary())

    return vecm_res

if __name__ == "__main__":
    print("TESTS DE STATIONNARITÉ ET COINTÉGRATION")
    test_stationarity_and_cointegration()

    MODE = "VECM"  # ⇦ Choisis entre "VAR" ou "VECM"

    if MODE == "VAR":
        print("Chargement des données et calcul des log-rendements mensuels...")
        log_returns = prepare_log_returns()

        print("Estimation du modèle VAR...")
        var_results = estimate_var_model(log_returns, lags=2)

        print("Affichage des prévisions en échantillon...")
        plot_in_sample_forecast(var_results)

    elif MODE == "VECM":
        print("Estimation du modèle VECM...")
        vecm_results = estimate_vecm_model()