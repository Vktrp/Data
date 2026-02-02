# 🏥 Système de Prédiction des Admissions - Pitié-Salpêtrière

Prédiction des admissions hospitalières à 7 jours utilisant Machine Learning et visualisations interactives.

## 🎯 Objectif

Anticiper les admissions pour optimiser la gestion des ressources (lits, personnel, stocks) et activer le Plan Blanc de manière préventive.

## 📊 Performances

- **Modèle** : Gradient Boosting (sélectionné parmi 3 algorithmes)
- **R² = 0.88** (88% de variance expliquée)
- **MAE = 2.28 patients** (erreur moyenne très faible)
- **MAPE = 6%**

## 🚀 Utilisation Rapide

### Lancer le dashboard
```bash
streamlit run dashboard.py
```

### Générer les prédictions
```bash
python3 models_comparison.py    # Compare XGBoost, Random Forest, Gradient Boosting
python3 model_prediction.py     # Génère previsions_future.csv
```

### Créer les visualisations
```bash
python3 graph.py               # Génère 4 graphiques d'analyse
```

## 📁 Structure

```
├── generateur/                # Scripts de génération de données (80k patients)
│   ├── patientsGenerator.py
│   ├── admissionsDailyGenerator.py
│   └── ...
├── models_comparison.py       # Comparaison des modèles ML
├── model_prediction.py        # Prédictions 7 jours
├── dashboard.py              # Interface Streamlit interactive
├── graph.py                  # Visualisations (graphs 1-4)
└── *.csv                     # Données générées
```

## 📈 Fonctionnalités Dashboard

✅ 4 KPI temps réel (admissions, lits, prévisions, risque)  
✅ Mode Plan Blanc (1800 → 2500 lits)  
✅ 3 onglets : Admissions / Occupation / Prévisions  
✅ Prévisions 7 jours avec intervalles de confiance  
✅ Alertes automatiques de saturation

## 📊 Graphiques Générés

- **graph1** : Admissions + événements (grippe, COVID, canicule)
- **graph2** : Saturation des lits (taux d'occupation)
- **graph3** : Tension sur le personnel
- **graph4** : Gestion des stocks (masques)
- **graphA** : Comparaison des 3 modèles ML
- **graphB** : Performance du modèle final
- **graphC** : Prévisions 7 jours
