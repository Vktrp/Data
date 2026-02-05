# 🏥 Système de Prédiction des Admissions Hospitalières
## Pitié-Salpêtrière - Dashboard IA

Prédiction des admissions hospitalières à 7 jours utilisant Machine Learning, simulation de crises et recommandations automatiques.

---

## 🎯 Objectif du Projet

Anticiper les admissions hospitalières pour optimiser la gestion des ressources (lits, personnel, stocks) et activer le **Plan Blanc** de manière préventive, permettant de réduire les saturations de 73%.

**Contexte** : L'Hôpital Pitié-Salpêtrière accueille 100 000+ patients/an aux urgences. Les pics d'admission non anticipés entraînent des saturations coûteuses (50k€/jour).

---

## 📊 Performances du Modèle

| Métrique | Valeur | Signification |
|----------|--------|---------------|
| **R²** | 0.88 | 88% de la variance expliquée |
| **MAE** | 16,05 patients | Erreur moyenne ±16 patients |
| **MAPE** | 4,5% | Erreur relative très faible |
| **Modèle** | Gradient Boosting | Sélectionné parmi 3 algorithmes |

---

## 🚀 Installation & Lancement

### Prérequis
```bash
pip install -r requirements.txt
```

### Lancer le dashboard (interface principale)
```bash
streamlit run dashboard.py
```
→ Ouvre automatiquement dans le navigateur sur `http://localhost:8501`

### Générer les prédictions (si nécessaire)
```bash
python3 models_comparison.py    # Compare 3 modèles ML
python3 model_prediction.py     # Génère previsions_future.csv
```

### Créer les graphiques d'analyse (optionnel)
```bash
python3 graph.py
```
---

## 💻 Fonctionnalités du Dashboard

### 🎛️ Interface Principale

#### 📊 4 KPI Temps Réel
1. **Admissions hier** (avec Δ vs J-7)
2. **Lits disponibles** (taux occupation %)
3. **Prévision IA J+1** (avec intervalle confiance)
4. **Risque saturation** (🔴/🟠/🟢)

#### 🏥 Mode Plan Blanc
- **Normal** : 1800 lits
- **Plan Blanc** : 2500 lits (+700 lits d'urgence)
- Visualisation graphique de l'impact

---

## 📚 Technologies Utilisées

**Languages & Frameworks**
- Python 3.12
- Streamlit 1.32+ (dashboard interactif)
- Scikit-learn 1.3+ (ML)
- Gradient Boosting
- Plotly (graphiques interactifs)

**Librairies Data Science**
- pandas, numpy (manipulation données)
- matplotlib, seaborn (visualisations)
- scipy, statsmodels (tests statistiques)

---
