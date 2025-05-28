import yaml
import yfinance as yf
import pandas as pd

def load_config(path: str = "config/config.yaml") -> dict:
    """Charge la configuration depuis un fichier YAML.
    Args: 
        path (str): Chemin vers le fichier de configuration YAML.
    Returns:
        dict: Configuration chargée.
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

def download_multiple_stocks(config: dict) -> pd.DataFrame:
    """ Télécharge les données boursières via l'API yahoo finance.
    Args:
        config (dict): Configuration contenant les paramètres de téléchargement.
    Returns:
        pd.DataFrame: DataFrame contenant les prix de clôture du S&P500 et de l'indice VIX.
    """
    tickers = config["yfinance"]["tickers"]
    start = config["yfinance"]["start_date"]
    end = config["yfinance"]["end_date"]
    interval = config["yfinance"]["interval"]

    data = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True)

    # Sécurisation + renommage propre
    vix = data["Close"].get("^VIX", pd.Series()).rename("VIX_Close")
    gspc = data["Close"].get("^GSPC", pd.Series()).rename("GSPC_Close")

    df = pd.concat([vix, gspc], axis=1)
    df.index.name = "Date"
    df.reset_index(inplace=True)

    return df

if __name__ == "__main__":
    config = load_config()
    df = download_multiple_stocks(config)
    print(df.head())