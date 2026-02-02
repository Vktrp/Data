import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

"""
ENTRAÎNEMENT DU MODÈLE FINAL
Utilise le meilleur modèle identifié par la comparaison
"""

sns.set_theme(style="whitegrid")

print("="*70)
print("🎯 ENTRAÎNEMENT DU MODÈLE FINAL")
print("="*70)

# =============================================================================
# 📂 CHARGEMENT
# =============================================================================

print("\n📂 Chargement des données...")
df = pd.read_csv("admissions_daily.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"✅ {len(df)} jours de données")

# Charger le nom du meilleur modèle (si existe)
try:
    with open("meilleur_modele.txt", "r") as f:
        best_model_name = f.read().strip()
    print(f"📌 Modèle sélectionné: {best_model_name}")
except FileNotFoundError:
    best_model_name = "Gradient Boosting"
    print(f"⚠️  Fichier meilleur_modele.txt non trouvé")
    print(f"📌 Utilisation du modèle par défaut: {best_model_name}")

# =============================================================================
# 🔧 FEATURE ENGINEERING
# =============================================================================

print("\n🔧 Création des features...")

# Features temporelles
df['jour_semaine'] = df['date'].dt.dayofweek
df['jour_mois'] = df['date'].dt.day
df['jour_annee'] = df['date'].dt.dayofyear
df['semaine_annee'] = df['date'].dt.isocalendar().week
df['mois'] = df['date'].dt.month
df['trimestre'] = df['date'].dt.quarter

df['is_monday'] = (df['jour_semaine'] == 0).astype(int)
df['is_tuesday'] = (df['jour_semaine'] == 1).astype(int)
df['is_friday'] = (df['jour_semaine'] == 4).astype(int)
df['is_weekend'] = (df['jour_semaine'] >= 5).astype(int)
df['is_debut_mois'] = (df['jour_mois'] <= 7).astype(int)
df['is_fin_mois'] = (df['jour_mois'] >= 24).astype(int)
df['is_event'] = df['event'].apply(lambda x: 1 if x != 'none' and pd.notnull(x) else 0)

df['sin_semaine'] = np.sin(2 * np.pi * df['jour_semaine'] / 7)
df['cos_semaine'] = np.cos(2 * np.pi * df['jour_semaine'] / 7)
df['sin_mois'] = np.sin(2 * np.pi * df['jour_mois'] / 31)
df['cos_mois'] = np.cos(2 * np.pi * df['jour_mois'] / 31)
df['sin_annee'] = np.sin(2 * np.pi * df['jour_annee'] / 365)
df['cos_annee'] = np.cos(2 * np.pi * df['jour_annee'] / 365)

for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]:
    df[f'lag_{lag}'] = df['nb_admissions'].shift(lag)

df['diff_1'] = df['nb_admissions'].diff(1)
df['diff_7'] = df['nb_admissions'].diff(7)

windows = [3, 7, 14, 21, 30]
for window in windows:
    df[f'rolling_mean_{window}'] = df['nb_admissions'].shift(1).rolling(window=window).mean()
    df[f'rolling_std_{window}'] = df['nb_admissions'].shift(1).rolling(window=window).std()
    df[f'rolling_min_{window}'] = df['nb_admissions'].shift(1).rolling(window=window).min()
    df[f'rolling_max_{window}'] = df['nb_admissions'].shift(1).rolling(window=window).max()

df['trend'] = df['rolling_mean_7'] - df['rolling_mean_30']
df['momentum_3'] = df['nb_admissions'].shift(1) - df['nb_admissions'].shift(4)
df['ratio_to_mean_7'] = df['nb_admissions'].shift(1) / (df['rolling_mean_7'] + 1)
df['monday_x_lag1'] = df['is_monday'] * df['lag_1']
df['weekend_x_mean7'] = df['is_weekend'] * df['rolling_mean_7']

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

features = [col for col in df.columns if col not in ['date', 'nb_admissions', 'event']]
target = 'nb_admissions'

print(f"✅ {len(features)} features créées")

# =============================================================================
# 📊 SPLIT
# =============================================================================

train_size = len(df) - 60
train = df.iloc[:train_size]
test = df.iloc[train_size:]

print(f"📊 Split: {len(train)} train / {len(test)} test")

# =============================================================================
# 🤖 SÉLECTION ET ENTRAÎNEMENT DU MODÈLE
# =============================================================================

print(f"\n{'='*70}")
print(f"🤖 SÉLECTION DU MODÈLE")
print(f"{'='*70}")
print(f"✅ Modèle choisi: {best_model_name}")
print(f"{'='*70}")

print(f"\n🏋️ Entraînement en cours...")

# Définir le modèle selon le choix
if best_model_name == "XGBoost":
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
elif best_model_name == "Random Forest":
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
else:  # Gradient Boosting
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    print(f"✅ Gradient Boosting Regressor initialisé")
    print(f"   Paramètres: n_estimators=500, learning_rate=0.05, max_depth=5")

print(f"\n📊 Entraînement sur {len(train)} observations...")

# Entraînement
model.fit(train[features], train[target])

# Prédictions
predictions = model.predict(test[features])

# Métriques
mae = mean_absolute_error(test[target], predictions)
rmse = np.sqrt(mean_squared_error(test[target], predictions))
r2 = r2_score(test[target], predictions)
mape = np.mean(np.abs((test[target] - predictions) / test[target])) * 100

print("\n" + "="*70)
print("📊 PERFORMANCE DU MODÈLE")
print("="*70)
print(f"Modèle: {best_model_name}")
print(f"MAE  : ±{mae:.2f} patients")
print(f"RMSE : ±{rmse:.2f} patients")
print(f"R²   : {r2:.4f} ({r2*100:.1f}% de variance expliquée)")
print(f"MAPE : {mape:.2f}%")
print("="*70)

# =============================================================================
# 📊 GRAPHIQUE DE PERFORMANCE
# =============================================================================

print("\n📊 Génération du graphique...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 1. Prédictions vs Réalité
ax1 = axes[0]
ax1.plot(test['date'], test[target], label='Réalité', 
         color='#2C3E50', linewidth=2.5, marker='o', markersize=5)
ax1.plot(test['date'], predictions, label=f'{best_model_name}', 
         color='#E74C3C', linestyle='--', linewidth=2.5, marker='s', markersize=5)
ax1.fill_between(test['date'], predictions - mae, predictions + mae,
                  alpha=0.2, color='#E74C3C', label=f'Intervalle ±{mae:.1f}')
ax1.set_title(f'🎯 {best_model_name} | MAE: {mae:.2f} | R²: {r2:.4f} | MAPE: {mape:.2f}%', 
              fontsize=14, fontweight='bold')
ax1.set_ylabel('Admissions', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# 2. Résidus
ax2 = axes[1]
residuals = test[target].values - predictions
colors = ['#27AE60' if r >= 0 else '#E74C3C' for r in residuals]
ax2.bar(range(len(residuals)), residuals, color=colors, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(y=mae, color='orange', linestyle='--', linewidth=1.5, label=f'MAE: ±{mae:.2f}')
ax2.axhline(y=-mae, color='orange', linestyle='--', linewidth=1.5)
ax2.set_title('📉 Analyse des Erreurs (Résidus)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Erreur', fontsize=11)
ax2.set_xlabel('Observation', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("graphB_modele_final.png", dpi=150, bbox_inches='tight')
plt.close()

print("✅ graphB_modele_final.png")

# =============================================================================
# 🔮 PRÉDICTIONS FUTURES (7 jours)
# =============================================================================

print("\n🔮 Calcul des prévisions futures...")

last_date = df['date'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
history_full = df[['date', 'nb_admissions']].copy()
future_preds = []
future_uncertainties = []

for i, date in enumerate(future_dates):
    temp_df = pd.concat([history_full, pd.DataFrame({'date': [date], 'nb_admissions': [np.nan]})], 
                        ignore_index=True)
    temp_df = temp_df.sort_values('date').reset_index(drop=True)
    
    # Recalculer features
    temp_df['jour_semaine'] = temp_df['date'].dt.dayofweek
    temp_df['jour_mois'] = temp_df['date'].dt.day
    temp_df['jour_annee'] = temp_df['date'].dt.dayofyear
    temp_df['semaine_annee'] = temp_df['date'].dt.isocalendar().week
    temp_df['mois'] = temp_df['date'].dt.month
    temp_df['trimestre'] = temp_df['date'].dt.quarter
    
    temp_df['is_monday'] = (temp_df['jour_semaine'] == 0).astype(int)
    temp_df['is_tuesday'] = (temp_df['jour_semaine'] == 1).astype(int)
    temp_df['is_friday'] = (temp_df['jour_semaine'] == 4).astype(int)
    temp_df['is_weekend'] = (temp_df['jour_semaine'] >= 5).astype(int)
    temp_df['is_debut_mois'] = (temp_df['jour_mois'] <= 7).astype(int)
    temp_df['is_fin_mois'] = (temp_df['jour_mois'] >= 24).astype(int)
    temp_df['is_event'] = 0
    
    temp_df['sin_semaine'] = np.sin(2 * np.pi * temp_df['jour_semaine'] / 7)
    temp_df['cos_semaine'] = np.cos(2 * np.pi * temp_df['jour_semaine'] / 7)
    temp_df['sin_mois'] = np.sin(2 * np.pi * temp_df['jour_mois'] / 31)
    temp_df['cos_mois'] = np.cos(2 * np.pi * temp_df['jour_mois'] / 31)
    temp_df['sin_annee'] = np.sin(2 * np.pi * temp_df['jour_annee'] / 365)
    temp_df['cos_annee'] = np.cos(2 * np.pi * temp_df['jour_annee'] / 365)
    
    for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]:
        temp_df[f'lag_{lag}'] = temp_df['nb_admissions'].shift(lag)
    
    temp_df['diff_1'] = temp_df['nb_admissions'].diff(1)
    temp_df['diff_7'] = temp_df['nb_admissions'].diff(7)
    
    for window in [3, 7, 14, 21, 30]:
        temp_df[f'rolling_mean_{window}'] = temp_df['nb_admissions'].shift(1).rolling(window=window).mean()
        temp_df[f'rolling_std_{window}'] = temp_df['nb_admissions'].shift(1).rolling(window=window).std()
        temp_df[f'rolling_min_{window}'] = temp_df['nb_admissions'].shift(1).rolling(window=window).min()
        temp_df[f'rolling_max_{window}'] = temp_df['nb_admissions'].shift(1).rolling(window=window).max()
    
    temp_df['trend'] = temp_df['rolling_mean_7'] - temp_df['rolling_mean_30']
    temp_df['momentum_3'] = temp_df['nb_admissions'].shift(1) - temp_df['nb_admissions'].shift(4)
    temp_df['ratio_to_mean_7'] = temp_df['nb_admissions'].shift(1) / (temp_df['rolling_mean_7'] + 1)
    temp_df['monday_x_lag1'] = temp_df['is_monday'] * temp_df['lag_1']
    temp_df['weekend_x_mean7'] = temp_df['is_weekend'] * temp_df['rolling_mean_7']
    
    temp_df = temp_df.replace([np.inf, -np.inf], np.nan)
    
    # Extraire et gérer les NaN
    current_row = temp_df.iloc[-1][features].copy()
    
    # Remplacer les NaN par les médianes du train
    for col in features:
        if pd.isna(current_row[col]):
            train_median = train[col].median()
            current_row[col] = train_median if not pd.isna(train_median) else 0
    
    # Prédiction
    pred = model.predict(current_row.values.reshape(1, -1))[0]
    
    future_preds.append(pred)
    future_uncertainties.append(mae * (1 + 0.15 * i))
    history_full = pd.concat([history_full, pd.DataFrame({'date': [date], 'nb_admissions': [pred]})], 
                             ignore_index=True)

# Sauvegarde
df_pred = pd.DataFrame({
    'date': future_dates,
    'pred_admissions': np.round(future_preds, 2),
    'pred_min': np.round(np.array(future_preds) - np.array(future_uncertainties), 2),
    'pred_max': np.round(np.array(future_preds) + np.array(future_uncertainties), 2)
})
df_pred.to_csv("previsions_future.csv", index=False)

# Graphique prévisions
fig2, ax = plt.subplots(figsize=(14, 7))
recent = df.tail(60)
ax.plot(recent['date'], recent['nb_admissions'], 
        label='Historique récent', color='#34495E', linewidth=2)
ax.plot(df_pred['date'], df_pred['pred_admissions'], 
        label=f'Prévisions {best_model_name}', color='#E74C3C', 
        linestyle='--', linewidth=3, marker='o', markersize=10)
ax.fill_between(df_pred['date'], df_pred['pred_min'], df_pred['pred_max'],
                alpha=0.3, color='#E74C3C', label='Intervalle de confiance')

for _, row in df_pred.iterrows():
    ax.text(row['date'], row['pred_admissions'] + 3, f"{row['pred_admissions']:.0f}", 
            ha='center', fontsize=10, fontweight='bold', color='#E74C3C')

ax.set_title(f'🔮 Prévisions 7 jours - {best_model_name}', fontsize=16, fontweight='bold')
ax.set_ylabel('Admissions', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graphC_previsions_7jours.png", dpi=150, bbox_inches='tight')
plt.close()

print("✅ graphC_previsions_7jours.png")

# =============================================================================
# 📋 RÉSUMÉ
# =============================================================================

print("\n" + "="*70)
print("🔮 PRÉVISIONS DES 7 PROCHAINS JOURS")
print("="*70)
for _, row in df_pred.iterrows():
    print(f"{row['date'].strftime('%Y-%m-%d (%A)')}: "
          f"{row['pred_admissions']:5.0f} patients [{row['pred_min']:5.0f}-{row['pred_max']:5.0f}]")

print("\n" + "="*70)
print("✅ ENTRAÎNEMENT TERMINÉ")
print("="*70)
print(f"\n📁 Fichiers générés:")
print(f"   - previsions_future.csv")
print(f"   - graphB_modele_final.png")
print(f"   - graphC_previsions_7jours.png")
print(f"\n🎯 Modèle prêt pour utilisation: {best_model_name}")
print(f"   Performance: MAE={mae:.2f}, R²={r2:.4f}")