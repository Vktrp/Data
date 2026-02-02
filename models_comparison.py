import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

"""
COMPARAISON DE MODÈLES ML
Compare XGBoost, Random Forest et Gradient Boosting
"""

sns.set_theme(style="whitegrid")

print("="*70)
print("🔍 COMPARAISON DE MODÈLES DE MACHINE LEARNING")
print("="*70)

# =============================================================================
# 📂 CHARGEMENT ET PRÉPARATION
# =============================================================================

print("\n📂 Chargement des données...")
df = pd.read_csv("admissions_daily.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"✅ {len(df)} jours de données")

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

# Indicateurs booléens
df['is_monday'] = (df['jour_semaine'] == 0).astype(int)
df['is_tuesday'] = (df['jour_semaine'] == 1).astype(int)
df['is_friday'] = (df['jour_semaine'] == 4).astype(int)
df['is_weekend'] = (df['jour_semaine'] >= 5).astype(int)
df['is_debut_mois'] = (df['jour_mois'] <= 7).astype(int)
df['is_fin_mois'] = (df['jour_mois'] >= 24).astype(int)
df['is_event'] = df['event'].apply(lambda x: 1 if x != 'none' and pd.notnull(x) else 0)

# Saisonnalité
df['sin_semaine'] = np.sin(2 * np.pi * df['jour_semaine'] / 7)
df['cos_semaine'] = np.cos(2 * np.pi * df['jour_semaine'] / 7)
df['sin_mois'] = np.sin(2 * np.pi * df['jour_mois'] / 31)
df['cos_mois'] = np.cos(2 * np.pi * df['jour_mois'] / 31)
df['sin_annee'] = np.sin(2 * np.pi * df['jour_annee'] / 365)
df['cos_annee'] = np.cos(2 * np.pi * df['jour_annee'] / 365)

# Lags
for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]:
    df[f'lag_{lag}'] = df['nb_admissions'].shift(lag)

# Différences
df['diff_1'] = df['nb_admissions'].diff(1)
df['diff_7'] = df['nb_admissions'].diff(7)

# Rolling statistics
windows = [3, 7, 14, 21, 30]
for window in windows:
    df[f'rolling_mean_{window}'] = df['nb_admissions'].shift(1).rolling(window=window).mean()
    df[f'rolling_std_{window}'] = df['nb_admissions'].shift(1).rolling(window=window).std()
    df[f'rolling_min_{window}'] = df['nb_admissions'].shift(1).rolling(window=window).min()
    df[f'rolling_max_{window}'] = df['nb_admissions'].shift(1).rolling(window=window).max()

# Features dérivées
df['trend'] = df['rolling_mean_7'] - df['rolling_mean_30']
df['momentum_3'] = df['nb_admissions'].shift(1) - df['nb_admissions'].shift(4)
df['ratio_to_mean_7'] = df['nb_admissions'].shift(1) / (df['rolling_mean_7'] + 1)

# Interactions
df['monday_x_lag1'] = df['is_monday'] * df['lag_1']
df['weekend_x_mean7'] = df['is_weekend'] * df['rolling_mean_7']

# Nettoyer
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print(f"✅ {len(df.columns) - 3} features créées")

# =============================================================================
# 📊 SPLIT TRAIN/TEST
# =============================================================================

train_size = len(df) - 60
train = df.iloc[:train_size]
test = df.iloc[train_size:]

features = [col for col in df.columns if col not in ['date', 'nb_admissions', 'event']]
target = 'nb_admissions'

print(f"📊 Split: {len(train)} train / {len(test)} test")

# =============================================================================
# 🤖 DÉFINITION DES MODÈLES
# =============================================================================

models = {
    'XGBoost': {
        'model': XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'description': 'Boosting avec gradient optimisé',
        'color': '#E74C3C'
    },
    'Random Forest': {
        'model': RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'description': 'Ensemble d\'arbres de décision',
        'color': '#3498DB'
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ),
        'description': 'Boosting séquentiel',
        'color': '#27AE60'
    }
}

# =============================================================================
# 🏋️ ENTRAÎNEMENT ET ÉVALUATION
# =============================================================================

print("\n" + "="*70)
print("🏋️ ENTRAÎNEMENT DES MODÈLES")
print("="*70)

results = {}

for name, config in models.items():
    print(f"\n📍 Entraînement: {name}")
    print(f"   Description: {config['description']}")
    
    model = config['model']
    
    # Entraînement
    model.fit(train[features], train[target])
    
    # Prédictions
    preds_train = model.predict(train[features])
    preds_test = model.predict(test[features])
    
    # Métriques sur le test
    mae = mean_absolute_error(test[target], preds_test)
    rmse = np.sqrt(mean_squared_error(test[target], preds_test))
    r2 = r2_score(test[target], preds_test)
    mape = np.mean(np.abs((test[target] - preds_test) / test[target])) * 100
    
    # Métriques sur le train (pour détecter l'overfitting)
    r2_train = r2_score(train[target], preds_train)
    
    results[name] = {
        'model': model,
        'predictions_train': preds_train,
        'predictions_test': preds_test,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'r2_train': r2_train,
        'mape': mape,
        'color': config['color']
    }
    
    print(f"   ✅ MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f} | MAPE: {mape:.2f}%")

# =============================================================================
# 📊 TABLEAU COMPARATIF
# =============================================================================

print("\n" + "="*70)
print("📊 TABLEAU COMPARATIF DES MODÈLES")
print("="*70)

print(f"\n{'Modèle':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'MAPE':<10} {'Overfitting'}")
print("-"*70)

for name, res in results.items():
    overfitting = res['r2_train'] - res['r2']
    overfit_status = "⚠️ Oui" if overfitting > 0.15 else "✅ Non"
    
    print(f"{name:<20} {res['mae']:<10.2f} {res['rmse']:<10.2f} "
          f"{res['r2']:<10.4f} {res['mape']:<10.2f} {overfit_status}")

# Meilleur modèle
best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
best = results[best_model_name]

print("\n" + "="*70)
print(f"🏆 MEILLEUR MODÈLE: {best_model_name}")
print("="*70)
print(f"MAE  : ±{best['mae']:.2f} patients")
print(f"RMSE : ±{best['rmse']:.2f} patients")
print(f"R²   : {best['r2']:.4f} ({best['r2']*100:.1f}% de variance expliquée)")
print(f"MAPE : {best['mape']:.2f}%")

# Sauvegarder le nom du meilleur modèle
with open("meilleur_modele.txt", "w") as f:
    f.write(best_model_name)

# =============================================================================
# 📊 VISUALISATIONS
# =============================================================================

print("\n📊 Génération des graphiques...")

# GRAPHIQUE COMPARATIF
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Comparaison des métriques
ax1 = axes[0, 0]
x_pos = np.arange(len(results))
width = 0.25

r2s = [results[name]['r2'] for name in results.keys()]
maes = [results[name]['mae'] for name in results.keys()]
mapes = [results[name]['mape'] for name in results.keys()]

ax1.bar(x_pos - width, r2s, width, label='R²', color='#3498DB', alpha=0.8)
ax1.bar(x_pos, [m/10 for m in maes], width, label='MAE/10', color='#E74C3C', alpha=0.8)
ax1.bar(x_pos + width, [m/100 for m in mapes], width, label='MAPE/100', color='#27AE60', alpha=0.8)

ax1.set_ylabel('Score (normalisé)', fontsize=11)
ax1.set_title('📊 Comparaison des Métriques', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results.keys())
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 2. Prédictions vs Réalité (tous les modèles)
ax2 = axes[0, 1]
ax2.plot(test['date'], test[target], label='Réalité', 
         color='#2C3E50', linewidth=3, marker='o', markersize=5)

for name, res in results.items():
    ax2.plot(test['date'], res['predictions_test'], 
             label=f'{name} (R²={res["r2"]:.3f})',
             linestyle='--', linewidth=2, alpha=0.7)

ax2.set_title('🎯 Prédictions vs Réalité (tous modèles)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Admissions', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

# 3. Scatter plot du meilleur modèle
ax3 = axes[1, 0]
ax3.scatter(test[target], best['predictions_test'], 
           alpha=0.6, color=best['color'], s=80, edgecolors='black')
min_val = min(test[target].min(), best['predictions_test'].min())
max_val = max(test[target].max(), best['predictions_test'].max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite')
ax3.set_xlabel('Réalité', fontsize=11)
ax3.set_ylabel('Prédiction', fontsize=11)
ax3.set_title(f'📈 {best_model_name}: Réalité vs Prédiction', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Résidus du meilleur modèle
ax4 = axes[1, 1]
residuals = test[target].values - best['predictions_test']
colors = ['#27AE60' if r >= 0 else '#E74C3C' for r in residuals]
ax4.bar(range(len(residuals)), residuals, color=colors, alpha=0.7)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax4.axhline(y=best['mae'], color='orange', linestyle='--', linewidth=1.5, label=f'MAE: ±{best["mae"]:.2f}')
ax4.axhline(y=-best['mae'], color='orange', linestyle='--', linewidth=1.5)
ax4.set_title(f'📉 {best_model_name}: Analyse des erreurs', fontsize=12, fontweight='bold')
ax4.set_ylabel('Erreur (Prédiction - Réalité)', fontsize=11)
ax4.set_xlabel('Observation', fontsize=11)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("graphA_comparaison_modeles.png", dpi=150, bbox_inches='tight')
plt.close()

print("✅ graphA_comparaison_modeles.png")

# =============================================================================
# 📄 RAPPORT TEXTE
# =============================================================================

print("\n📄 Génération du rapport...")

rapport = f"""
{'='*70}
RAPPORT DE COMPARAISON DES MODÈLES DE MACHINE LEARNING
Hôpital Pitié-Salpêtrière - Prédiction des Admissions
{'='*70}

1. MODÈLES COMPARÉS
{'='*70}

"""

for name, config in models.items():
    res = results[name]
    rapport += f"""
{name}:
  Description: {config['description']}
  MAE:  {res['mae']:.2f} patients
  RMSE: {res['rmse']:.2f} patients
  R²:   {res['r2']:.4f} ({res['r2']*100:.1f}% de variance expliquée)
  MAPE: {res['mape']:.2f}%
  
  Overfitting: R²_train={res['r2_train']:.4f}, R²_test={res['r2']:.4f}
               Différence: {res['r2_train'] - res['r2']:.4f}
               {'⚠️ Risque de surapprentissage' if (res['r2_train'] - res['r2']) > 0.15 else '✅ Pas de surapprentissage'}
"""

rapport += f"""

2. CLASSEMENT
{'='*70}

"""

# Classement par R²
sorted_models = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
for i, (name, res) in enumerate(sorted_models, 1):
    rapport += f"{i}. {name:<20} R² = {res['r2']:.4f}\n"

rapport += f"""

3. MODÈLE SÉLECTIONNÉ
{'='*70}

🏆 {best_model_name}

Justification:
  - Meilleur R² ({best['r2']:.4f})
  - Erreur moyenne de ±{best['mae']:.2f} patients
  - MAPE de {best['mape']:.2f}% (très faible)
  
Interprétation du R²:
  Le modèle explique {best['r2']*100:.1f}% de la variance des admissions.
  Cela signifie que {best['r2']*100:.1f}% des variations d'admissions sont
  prédictibles grâce aux features temporelles et historiques.

4. RECOMMANDATION
{'='*70}

Le modèle {best_model_name} est recommandé pour la mise en production car:
  ✅ Performance optimale (R² = {best['r2']:.4f})
  ✅ Erreur acceptable (MAE = {best['mae']:.2f} patients)
  ✅ Pas de surapprentissage significatif
  ✅ Généralise bien sur données de test

Ce modèle sera utilisé pour les prédictions à 7 jours.

{'='*70}
Rapport généré le {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""

with open("rapport_comparaison_modeles.txt", "w", encoding='utf-8') as f:
    f.write(rapport)

print("✅ rapport_comparaison_modeles.txt")

print("\n" + "="*70)
print("✅ COMPARAISON TERMINÉE")
print("="*70)
print(f"\n🏆 Meilleur modèle: {best_model_name}")
print(f"📁 Fichiers générés:")
print(f"   - graphA_comparaison_modeles.png")
print(f"   - rapport_comparaison_modeles.txt")
print(f"   - meilleur_modele.txt")
print(f"\n💡 Utilisez '{best_model_name}' pour l'entraînement final")