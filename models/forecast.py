import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

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
    results = model.fit(lags)
    print(results.summary())
    return results

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