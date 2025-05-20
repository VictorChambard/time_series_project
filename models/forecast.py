import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def test_stationarity_and_cointegration(data_path="data/processed/clean_data.csv"):
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # Log des prix (pas les rendements)
    log_prices = np.log(df[["VIX_Close", "GSPC_Close"]])

    print("TEST ADF (stationnarité des log-prix)")
    for col in log_prices.columns:
        result = adfuller(log_prices[col])
        print(f"{col}: p-value = {result[1]:.4f} → {'stationnaire' if result[1] < 0.05 else 'non stationnaire'}")

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

test_stationarity_and_cointegration()

def estimate_var_model(log_returns: pd.DataFrame, lags: int = 1):
    model = VAR(log_returns)
    lag_order = model.select_order(maxlags=12)
    print(lag_order.summary())
    optimal_lag = lag_order.selected_orders["aic"]
    results = model.fit(optimal_lag)

def plot_in_sample_forecast(results, steps=12):
    forecast = results.fittedvalues

    forecast.plot(figsize=(10, 4))
    plt.title("Prévision en échantillon (in-sample forecast)")
    plt.xlabel("Date")
    plt.ylabel("Log-rendements")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("📈 Chargement des données et calcul des log-rendements mensuels...")
    log_returns = prepare_log_returns()

    print("✅ Estimation du modèle VAR...")
    var_results = estimate_var_model(log_returns, lags=2)

    print("📊 Affichage des prévisions en échantillon...")
    plot_in_sample_forecast(var_results)