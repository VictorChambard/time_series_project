# Time Series Project – VIX & S&P 500

Projet Python réalisé dans le cadre du cours Applied Data Science in Finance  
Master 1 MBFA – Université Paris 1 Panthéon-Sorbonne  
Encadré par Fabrice Galan

---

## Objectif

Étudier la dynamique conjointe entre le VIX (indice de volatilité) et le S&P 500 à travers l'estimation de modèles VAR et VECM.  
Analyser la qualité prédictive, notamment en période de crise (COVID), et comparer les prévisions à la volatilité réalisée.

---

## Membres du projet

- Victor Chambard  
- Niama EL KAMAL  
- Youssef Saied

---

## Méthodes utilisées

- Modèles VAR / VECM
- Prévisions multivariées
- Fonctions de réponse impulsionnelle (IRF)
- Tests de stationnarité (ADF) et de co-intégration (Johansen)
- Analyse de la volatilité réalisée (rolling std)

---

## Technologies

- Python 3.11  
- pandas, numpy, statsmodels, matplotlib, streamlit, yfinance

---
## Bibliographie
- Zhang, B., Hu, Y., & Ji, Q. (2020). Financial markets under the global pandemic of COVID-19. Finance Research Letters, 36, 101528.
- Becker, R., Clements, A. E., & McClelland, A. (2022). Forecasting Realized Volatility and VIX: A comparison of VAR and Machine Learning Models.

---

## Lancer le projet


# Créer l’environnement virtuel
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows
pip install -r requirements.txt

# Lancer le dashboard
python main.py
