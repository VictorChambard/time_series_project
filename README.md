# Time Series Project – VAR sur VIX & S&P 500

Projet Python réalisé dans le cadre du cours **Applied Data Science in Finance – M1 MBFA Paris 1**  
Encadré par **Fabrice Galan**

---

## 🎯 Objectif

L’objectif de ce projet est d’appliquer un **modèle VAR (Vector AutoRegressive)** pour analyser la dynamique conjointe de :
- **l’indice de volatilité VIX**
- **l’indice boursier S&P 500**

Nous avons :
- Collecté les données financières avec `yfinance` (2004–2024)
- Nettoyé et structuré le jeu de données via un pipeline ETL complet
- Estimé un **VAR sur les log-rendements mensuels**
- Réalisé des prévisions en échantillon
- Structuré le projet pour faciliter la réutilisabilité (config, modules, tests)

---

## 🗂️ Structure du projet