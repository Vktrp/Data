import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import holidays
import time


# =============================================================================
# 1. PRÉPARATION DES DONNÉES
# =============================================================================

df = pd.read_csv("admissions_daily.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Features
fr_holidays = holidays.France()
df["is_holiday"] = df["date"].isin(fr_holidays).astype(int)
df["veille_holiday"] = df["date"].isin([d - pd.Timedelta(days=1) for d in fr_holidays if d in df["date"].values]).astype(int)
df["lendemain_holiday"] = df["date"].isin([d + pd.Timedelta(days=1) for d in fr_holidays if d in df["date"].values]).astype(int)

# Gestion Covid
df["is_covid"] = 0
df.loc[(df["date"] >= "2020-03-01") & (df["date"] <= "2021-06-30"), "is_covid"] = 1

# Temporel
df["jour_semaine"] = df["date"].dt.dayofweek
df["mois"] = df["date"].dt.month
df["jour_annee"] = df["date"].dt.dayofyear
df["semaine_annee"] = df["date"].dt.isocalendar().week.astype(int)
df["annee"] = df["date"].dt.year
df["weekend"] = (df["jour_semaine"] >= 5).astype(int)
df["lundi"] = (df["jour_semaine"] == 0).astype(int)
df["decembre"] = (df["mois"] == 12).astype(int)
df["hiver"] = df["mois"].isin([11, 12, 1, 2]).astype(int)

# Cycles
df["sin_annee"] = np.sin(2 * np.pi * df["jour_annee"] / 365)
df["cos_annee"] = np.cos(2 * np.pi * df["jour_annee"] / 365)
df["sin_semaine"] = np.sin(2 * np.pi * df["jour_semaine"] / 7)
df["cos_semaine"] = np.cos(2 * np.pi * df["jour_semaine"] / 7)

# Lags & Rolling
for lag in [1, 7, 14, 21, 28, 364, 365]:
    df[f"lag_{lag}"] = df["nb_admissions"].shift(lag)

for w in [7, 14, 30, 60]:
    df[f"roll_mean_{w}"] = df["nb_admissions"].shift(1).rolling(w).mean()
    df[f"roll_std_{w}"] = df["nb_admissions"].shift(1).rolling(w).std()

df["trend_7_30"] = df["roll_mean_7"] - df["roll_mean_30"]

df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

features = [
    "jour_semaine", "semaine_annee", "weekend", "lundi", "decembre", "hiver", "annee",
    "is_holiday", "veille_holiday", "lendemain_holiday", "is_covid",
    "sin_annee", "cos_annee", "sin_semaine", "cos_semaine",
    "lag_1", "lag_7", "lag_14", "lag_21", "lag_28", "lag_364", "lag_365",
    "roll_mean_7", "roll_mean_14", "roll_mean_30", "roll_mean_60",
    "roll_std_7", "roll_std_30", "trend_7_30"
]


# =============================================================================
# 2. CONFIGURATION DES COMBATTANTS
# =============================================================================

models_config = {
    "XGBoost (Poisson)": XGBRegressor(
        objective="count:poisson",
        n_estimators=500, learning_rate=0.02, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, n_jobs=-1, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_split=10,
        n_jobs=-1, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        loss="absolute_error", random_state=42
    )
}

# =============================================================================
# 3. LA COMPÉTITION 
# =============================================================================

n_weeks = 5
horizons = range(1, 8) # J+1 à J+7
results = {}

print(f"\n🥊 DÉBUT DU MATCH (Test sur {n_weeks} semaines glissantes)...")

for name, model_template in models_config.items():
    print(f"\n🔹 Test de {name}...", end=" ")
    start_time = time.time()
    
    errors = []
    
    # Point de départ du test
    min_train = len(df) - (n_weeks * 7) - 60
    
    for w in range(n_weeks):
        train_end = min_train + w * 7
        train_df = df.iloc[:train_end].copy()
        
        # Pour chaque horizon, on entraîne un modèle spécifique (Multi-Horizon Direct)
        for h in horizons:
            train_df[f"y_h{h}"] = train_df["nb_admissions"].shift(-h)
            tmp = train_df.dropna()
            
            # Entraînement
            # On clone le modèle pour repartir de zéro à chaque fois
            if "XGB" in name:
                model = XGBRegressor(**model_template.get_params())
            elif "Random" in name:
                model = RandomForestRegressor(**model_template.get_params())
            else:
                model = GradientBoostingRegressor(**model_template.get_params())
                
            model.fit(tmp[features], tmp[f"y_h{h}"])
            
            # Prédiction
            if train_end + h < len(df):
                feat_row = df.iloc[[train_end]][features]
                pred = model.predict(feat_row)[0]
                true = df.iloc[train_end + h]["nb_admissions"]
                errors.append(abs(pred - true))
        
        print(".", end="", flush=True)
        
    # Calcul des scores
    mae = np.mean(errors)
    duration = time.time() - start_time
    results[name] = {"MAE": mae, "Time": duration}
    print(f" Terminé ({duration:.1f}s) -> MAE: {mae:.2f}")

# =============================================================================
# 4. RESULTAT
# =============================================================================


# Tri par MAE croissant (le plus petit gagne)
sorted_results = dict(sorted(results.items(), key=lambda item: item[1]['MAE']))

print(f"{'Rang':<6} {'Modèle':<25} {'MAE (Erreur)':<15} {'Temps Calcul'}")
print("-" * 60)

for rank, (name, metrics) in enumerate(sorted_results.items(), 1):
    print(f"{rank:<6} {name:<25} {metrics['MAE']:<15.2f} {metrics['Time']:.1f}s")

winner = list(sorted_results.keys())[0]

# --- SAUVEGARDE DU MEILLEUR MODÈLE ---
with open("meilleur_modele.txt", "w") as f:
    f.write(winner)
print(f"✅ Nom du vainqueur sauvegardé dans 'meilleur_modele.txt'")

# =============================================================================
# 5. GRAPHIQUE COMPARATIF
# =============================================================================

names = list(results.keys())
maes = [results[n]["MAE"] for n in names]
colors = ['#27AE60' if n == winner else '#95A5A6' for n in names]

plt.figure(figsize=(10, 6))
bars = plt.bar(names, maes, color=colors)
plt.title("Comparaison de l'Erreur Moyenne (MAE) - Méthode Walk-Forward")
plt.ylabel("Erreur Moyenne (Patients)")
plt.grid(axis='y', alpha=0.3)

# Ajouter les chiffres sur les barres
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

plt.savefig("graphA_comparaison_modeles.png")
print("\n📸 Graphique 'graphA_comparaison_modeles.png' généré.")