import matplotlib.pyplot as plt
from etl.extract import load_config, download_multiple_stocks
from etl.transform import clean_multivariate_data
from helpers.volatilite import calcul_volatilite  

config = load_config()
df = download_multiple_stocks(config)
df = clean_multivariate_data(df)

vol = calcul_volatilite(df["GSPC_Close"])

plt.figure(figsize=(10, 5))
plt.plot(df["Date"], vol)
plt.title("Volatilité réalisée annualisée (S&P500)")
plt.xlabel("Date")
plt.ylabel("Volatilité")
plt.grid(True)
plt.tight_layout()
plt.show()
