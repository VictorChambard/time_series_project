import matplotlib
matplotlib.use('Agg')
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import matplotlib.pyplot as plt
from etl.extract import load_config, download_multiple_stocks
from etl.transform import clean_multivariate_data
from helpers.volatilite import calcul_volatilite  

config = load_config()
df = download_multiple_stocks(config)
df = clean_multivariate_data(df)

vol = calcul_volatilite(df["GSPC_Close"])

plt.figure(figsize=(10, 5))
masque_valide = vol.notna()
plt.plot(df["Date"][masque_valide], vol[masque_valide])
plt.title("Volatilité réalisée annualisée (S&P500)")
plt.xlabel("Date")
plt.ylabel("Volatilité")
plt.grid(True)
plt.tight_layout()
plt.show()
