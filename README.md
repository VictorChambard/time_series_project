# Time Series Project – VIX & S&P 500

Projet Python réalisé dans le cadre du cours **Applied Data Science in Finance – M1 MBFA Paris 1**, encadré par **Fabrice Galan**.

---

## 🎯 Objectif

Mettre en œuvre un pipeline complet de traitement de données financières et estimer un modèle **VAR (Vector AutoRegressive)** sur les **log-rendements mensuels** du **VIX** et du **S&P 500** (2004–2024).

---

## ⚙️ Technologies utilisées

- Python 3.11
- `pandas`, `numpy`, `statsmodels`, `yfinance`
- `matplotlib` (ou `plotly`)
- `yaml`, `Streamlit` (à venir)

---

## ▶️ Lancer le projet

```bash
python -m venv venv
source venv\Scripts\activate 
pip install -r requirements.txt

# Lancer le pipeline ETL
python main.py

# Estimer le VAR
python models/forecast.py