import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ON CHANGE D'ALGORITHME ICI
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import holidays
import pickle

print("=" * 80)
print("🎯 MODÈLE FINAL : ADMISSIONS URGENCES (GRADIENT BOOSTING)")
print("=" * 80)

# =============================================================================
# 📂 DONNÉES
# =============================================================================

df = pd.read_csv("admissions_daily.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"✅ {len(df)} jours chargés")

# =============================================================================
# 🔧 FEATURES (IDENTIQUES AU TEST COMPARATIF)
# =============================================================================

fr_holidays = holidays.France()

# 1. Gestion Jours Fériés & Vacances
df["is_holiday"] = df["date"].isin(fr_holidays).astype(int)
df["veille_holiday"] = df["date"].isin(
    [d - pd.Timedelta(days=1) for d in fr_holidays if d in df["date"].values]
).astype(int)
df["lendemain_holiday"] = df["date"].isin(
    [d + pd.Timedelta(days=1) for d in fr_holidays if d in df["date"].values]
).astype(int)

# 2. GESTION SPÉCIALE COVID
df["is_covid"] = 0
df.loc[(df["date"] >= "2020-03-01") & (df["date"] <= "2021-06-30"), "is_covid"] = 1

# 3. Temporel
df["jour_semaine"] = df["date"].dt.dayofweek
df["mois"] = df["date"].dt.month
df["jour_annee"] = df["date"].dt.dayofyear
df["semaine_annee"] = df["date"].dt.isocalendar().week.astype(int)
df["annee"] = df["date"].dt.year 

df["weekend"] = (df["jour_semaine"] >= 5).astype(int)
df["lundi"] = (df["jour_semaine"] == 0).astype(int)
df["decembre"] = (df["mois"] == 12).astype(int)
df["hiver"] = df["mois"].isin([11, 12, 1, 2]).astype(int)

# 4. Cycles
df["sin_annee"] = np.sin(2 * np.pi * df["jour_annee"] / 365)
df["cos_annee"] = np.cos(2 * np.pi * df["jour_annee"] / 365)
df["sin_semaine"] = np.sin(2 * np.pi * df["jour_semaine"] / 7)
df["cos_semaine"] = np.cos(2 * np.pi * df["jour_semaine"] / 7)

# 5. Lags
for lag in [1, 7, 14, 21, 28, 364, 365]: 
    df[f"lag_{lag}"] = df["nb_admissions"].shift(lag)

# 6. Moyennes Glissantes
for w in [7, 14, 30, 60]:
    df[f"roll_mean_{w}"] = df["nb_admissions"].shift(1).rolling(w).mean()
    df[f"roll_std_{w}"] = df["nb_admissions"].shift(1).rolling(w).std()

# 7. Tendance
df["trend_7_30"] = df["roll_mean_7"] - df["roll_mean_30"]

# Nettoyage
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

features = [
    "jour_semaine", "semaine_annee", "weekend", "lundi",
    "decembre", "hiver", "annee",
    "is_holiday", "veille_holiday", "lendemain_holiday",
    "is_covid",  
    "sin_annee", "cos_annee", "sin_semaine", "cos_semaine",
    "lag_1", "lag_7", "lag_14", "lag_21", "lag_28", "lag_364", "lag_365",
    "roll_mean_7", "roll_mean_14", "roll_mean_30", "roll_mean_60",
    "roll_std_7", "roll_std_30",
    "trend_7_30"
]

print(f"✅ {len(features)} features utilisées (dont gestion COVID)")

# =============================================================================
# 🧠 MODÈLE (GRADIENT BOOSTING - LE VAINQUEUR)
# =============================================================================

def make_model():
    # Configuration exacte qui a gagné le test (MAE 15.13)
    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        loss="absolute_error", # Optimise le MAE directement !
        random_state=42
    )

# =============================================================================
# 🧪 VALIDATION WALK-FORWARD
# =============================================================================

print("\n" + "="*80)
print("🧪 VALIDATION WALK-FORWARD")
print("="*80)

horizons = range(1, 8)
n_test_weeks = 10
errors = {h: [] for h in horizons}
all_predictions = {h: [] for h in horizons}
all_actuals = {h: [] for h in horizons}

min_train = len(df) - n_test_weeks * 7 - 100

print(f"\n🔄 Validation sur {n_test_weeks} semaines (Patience, Gradient Boosting est lent)...")

for w in range(n_test_weeks):
    train_end = min_train + w * 7
    train_df = df.iloc[:train_end].copy()
    
    print(".", end="", flush=True)

    for h in horizons:
        train_df[f"y_h{h}"] = train_df["nb_admissions"].shift(-h)
        tmp = train_df.dropna()

        model = make_model()
        model.fit(tmp[features], tmp[f"y_h{h}"])

        if train_end + h < len(df):
            y_true = df.iloc[train_end + h]["nb_admissions"]
            y_pred = model.predict(df.iloc[[train_end]][features])[0]

            errors[h].append(abs(y_pred - y_true))
            all_predictions[h].append(y_pred)
            all_actuals[h].append(y_true)

# =============================================================================
# 📊 RÉSULTATS VALIDATION
# =============================================================================

print("\n\n" + "="*80)
print("📊 RÉSULTATS VALIDATION")
print("="*80)

mae_by_h = [np.mean(errors[h]) for h in horizons if errors[h]]
mae_global = np.mean(mae_by_h)

# Calculer R² global
all_actuals_flat = [a for h in horizons for a in all_actuals[h]]
all_preds_flat = [p for h in horizons for p in all_predictions[h]]
r2_global = r2_score(all_actuals_flat, all_preds_flat)

print(f"\n🎯 Performance globale :")
print(f"   MAE global : {mae_global:.2f} patients")
print(f"   R² global  : {r2_global:.4f}")

# Comparaison baseline
if all_actuals_flat:
    baseline_pred = np.mean(all_actuals_flat)
    mae_baseline = np.mean([abs(a - baseline_pred) for a in all_actuals_flat])
    improvement = ((mae_baseline - mae_global) / mae_baseline) * 100
    print(f"   Amélioration baseline : {improvement:.1f}%")

# Graphique MAE
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(list(horizons), mae_by_h, marker="o", linewidth=2, markersize=8, color='#27AE60')
ax.axhline(y=mae_global, color='red', linestyle='--', label=f'Moyenne: {mae_global:.2f}')
ax.set_title(f"Erreur MAE par horizon (Gradient Boosting)", fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.savefig("mae_par_horizon.png")
plt.close()
print("✅ Graphique : mae_par_horizon.png")

# =============================================================================
# 🔮 ENTRAÎNEMENT FINAL + CSV + PRÉVISIONS
# =============================================================================

print("\n" + "="*80)
print("🔮 ENTRAÎNEMENT FINAL (Attention: ~2 minutes)")
print("="*80)

final_models = {}

# Entraînement des 7 modèles finaux
for h in horizons:
    print(f"   Entraînement horizon J+{h}...")
    df[f"y_h{h}"] = df["nb_admissions"].shift(-h)
    tmp = df.dropna()
    model = make_model()
    model.fit(tmp[features], tmp[f"y_h{h}"])
    final_models[h] = model

# Sauvegarde des modèles
with open("models_multi_horizon.pkl", "wb") as f:
    pickle.dump(final_models, f)
print("✅ Modèles sauvegardés : models_multi_horizon.pkl")

# Prédictions
last_row = df.iloc[-1]
future_dates = pd.date_range(start=last_row["date"] + pd.Timedelta(days=1), periods=7)
predictions = []

for h in horizons:
    pred = final_models[h].predict(last_row[features].values.reshape(1, -1))[0]
    predictions.append(pred)

# --- CRÉATION DU FICHIER CSV ---
df_final = pd.DataFrame({
    'date': future_dates,
    'pred_admissions': np.round(predictions, 0),
    'pred_min': np.round([p - mae_global for p in predictions], 0),
    'pred_max': np.round([p + mae_global for p in predictions], 0)
})

df_final.to_csv("previsions_future.csv", index=False)
print("✅ CSV GÉNÉRÉ : previsions_future.csv")

# Graphique Final
fig, ax = plt.subplots(figsize=(14, 7))
history_plot = df.tail(60)
ax.plot(history_plot["date"], history_plot["nb_admissions"], label="Historique", color='#2C3E50')
ax.plot(future_dates, predictions, marker="o", label="Prévisions", color='#E74C3C')
ax.fill_between(future_dates, df_final['pred_min'], df_final['pred_max'], color='#E74C3C', alpha=0.2, label=f"Marge ±{mae_global:.0f}")
ax.set_title(f"Prévisions 7 jours | MAE={mae_global:.2f}", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig("graphC_previsions_7jours.png")
plt.close()
print("✅ Graphique : graphC_previsions_7jours.png")

# Affichage terminal
print("\n" + "="*80)
print("📅 PRÉVISIONS SEMAINE PROCHAINE")
for _, row in df_final.iterrows():
    print(f"   {row['date'].strftime('%Y-%m-%d (%A)')}: {row['pred_admissions']:.0f} patients")
print("="*80)