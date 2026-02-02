import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

"""
ANALYSE STATISTIQUE COMPLÈTE
Génère graphiques 7-12 + rapport statistique
"""

print("="*70)
print("📊 ANALYSE STATISTIQUE COMPLÈTE")
print("="*70)

# Chargement
df = pd.read_csv("admissions_daily.csv", parse_dates=["date"])
df_patients = pd.read_csv("patients.csv", parse_dates=["date_admission"])

# =============================================================================
# TESTS STATISTIQUES
# =============================================================================

print("\n🔬 TESTS STATISTIQUES")
print("-"*70)

# 1. Test Student : Hiver vs Été
df['mois'] = df['date'].dt.month
hiver = df[df['mois'].isin([1, 2, 12])]['nb_admissions']
ete = df[df['mois'].isin([6, 7, 8])]['nb_admissions']

t_stat, p_value = stats.ttest_ind(hiver, ete)
print(f"\n1. Test t de Student : Hiver vs Été")
print(f"   H0 : Pas de différence entre hiver et été")
print(f"   Moyenne hiver : {hiver.mean():.1f}")
print(f"   Moyenne été   : {ete.mean():.1f}")
print(f"   t-statistic   : {t_stat:.3f}")
print(f"   p-value       : {p_value:.6f}")
print(f"   Conclusion    : {'Différence significative' if p_value < 0.05 else 'Pas de différence'}")

# 2. Corrélation Gravité vs Durée de séjour
corr = df_patients['gravite'].corr(df_patients['duree_sejour'])
print(f"\n2. Corrélation Gravité × Durée de séjour")
print(f"   Coefficient de Pearson : {corr:.3f}")
print(f"   R² : {corr**2:.3f} ({corr**2*100:.1f}% de variance expliquée)")

# 3. ANOVA : Durée de séjour par service
services = df_patients['service'].unique()
groups = [df_patients[df_patients['service'] == s]['duree_sejour'] for s in services]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"\n3. ANOVA : Durée de séjour par service")
print(f"   F-statistic : {f_stat:.3f}")
print(f"   p-value     : {p_anova:.6f}")
print(f"   Conclusion  : {'Différence significative entre services' if p_anova < 0.05 else 'Pas de différence'}")

# 4. Test normalité (Shapiro-Wilk)
stat_shapiro, p_shapiro = stats.shapiro(df['nb_admissions'].sample(min(5000, len(df))))
print(f"\n4. Test de Shapiro-Wilk : Normalité des admissions")
print(f"   Statistique : {stat_shapiro:.3f}")
print(f"   p-value     : {p_shapiro:.6f}")
print(f"   Conclusion  : {'Distribution normale' if p_shapiro > 0.05 else 'Distribution non-normale'}")

# =============================================================================
# GRAPH 7 : HEATMAP Jour × Mois
# =============================================================================

print("\n📊 Génération Graph 7 : Heatmap...")

df['jour_semaine'] = df['date'].dt.day_name()
df['mois_nom'] = df['date'].dt.month_name()

pivot = df.pivot_table(
    values='nb_admissions',
    index='jour_semaine',
    columns='mois',
    aggfunc='mean'
)

# Réordonner
jours_ordre = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot = pivot.reindex(jours_ordre)

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Admissions moyennes'})
ax.set_title('Heatmap : Admissions moyennes par Jour × Mois', fontsize=14, fontweight='bold')
ax.set_ylabel('Jour de la semaine', fontsize=11)
ax.set_xlabel('Mois', fontsize=11)
plt.tight_layout()
plt.savefig("graph7_heatmap_admissions.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ graph7_heatmap_admissions.png")

# =============================================================================
# GRAPH 8 : BOXPLOT par Mois
# =============================================================================

print("📊 Génération Graph 8 : Boxplot mensuel...")

fig, ax = plt.subplots(figsize=(12, 6))
df.boxplot(column='nb_admissions', by='mois', ax=ax, grid=False)
ax.set_title('Distribution des Admissions par Mois', fontsize=14, fontweight='bold')
ax.set_xlabel('Mois', fontsize=11)
ax.set_ylabel('Nombre d\'admissions', fontsize=11)
plt.suptitle('')  # Enlever titre auto
plt.tight_layout()
plt.savefig("graph8_boxplot_mois.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ graph8_boxplot_mois.png")

# =============================================================================
# GRAPH 9 : DÉCOMPOSITION Série Temporelle
# =============================================================================

print("📊 Génération Graph 9 : Décomposition série temporelle...")

df_ts = df.set_index('date')['nb_admissions']
decomposition = seasonal_decompose(df_ts, model='additive', period=7)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))

decomposition.observed.plot(ax=axes[0], color='#3498DB')
axes[0].set_ylabel('Observé', fontsize=10)
axes[0].set_title('Décomposition Série Temporelle', fontsize=14, fontweight='bold')

decomposition.trend.plot(ax=axes[1], color='#E74C3C')
axes[1].set_ylabel('Tendance', fontsize=10)

decomposition.seasonal.plot(ax=axes[2], color='#27AE60')
axes[2].set_ylabel('Saisonnalité', fontsize=10)

decomposition.resid.plot(ax=axes[3], color='#95A5A6')
axes[3].set_ylabel('Résidus', fontsize=10)
axes[3].set_xlabel('Date', fontsize=10)

plt.tight_layout()
plt.savefig("graph9_decomposition.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ graph9_decomposition.png")

# =============================================================================
# GRAPH 10 : CORRÉLATION Gravité × Durée
# =============================================================================

print("📊 Génération Graph 10 : Corrélation...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df_patients['gravite'], df_patients['duree_sejour'], 
          alpha=0.3, s=20, color='#3498DB')

# Régression
z = np.polyfit(df_patients['gravite'], df_patients['duree_sejour'], 1)
p = np.poly1d(z)
x_line = np.linspace(1, 5, 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

ax.set_xlabel('Gravité (1-5)', fontsize=11)
ax.set_ylabel('Durée de séjour (jours)', fontsize=11)
ax.set_title(f'Corrélation Gravité × Durée de séjour (r = {corr:.3f}, R² = {corr**2:.3f})', 
            fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graph10_correlation.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ graph10_correlation.png")

# =============================================================================
# GRAPH 11 : VIOLIN PLOT Durée par Service
# =============================================================================

print("📊 Génération Graph 11 : Violin plot...")

fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(data=df_patients, x='service', y='duree_sejour', ax=ax, palette='Set2')
ax.set_title('Distribution Durée de Séjour par Service', fontsize=14, fontweight='bold')
ax.set_xlabel('Service', fontsize=11)
ax.set_ylabel('Durée (jours)', fontsize=11)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graph11_distribution_sejour.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ graph11_distribution_sejour.png")

# =============================================================================
# GRAPH 12 : AUTOCORRÉLATION
# =============================================================================

print("📊 Génération Graph 12 : Autocorrélation...")

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(df['nb_admissions'], lags=30, ax=axes[0])
axes[0].set_title('ACF : Fonction d\'Autocorrélation', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Lag (jours)', fontsize=10)

plot_pacf(df['nb_admissions'], lags=30, ax=axes[1])
axes[1].set_title('PACF : Fonction d\'Autocorrélation Partielle', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Lag (jours)', fontsize=10)

plt.tight_layout()
plt.savefig("graph12_autocorrelation.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ graph12_autocorrelation.png")

# =============================================================================
# RAPPORT TEXTE
# =============================================================================

rapport = f"""
{'='*70}
RAPPORT D'ANALYSE STATISTIQUE
Hôpital Pitié-Salpêtrière
{'='*70}

1. TESTS STATISTIQUES
{'='*70}

1.1. Test t de Student : Hiver vs Été
   Hypothèse H0 : Pas de différence entre hiver et été
   Moyenne hiver : {hiver.mean():.1f} admissions/jour
   Moyenne été   : {ete.mean():.1f} admissions/jour
   Différence    : {(hiver.mean() - ete.mean()):.1f} (+{((hiver.mean()/ete.mean() - 1)*100):.1f}%)
   t-statistic   : {t_stat:.3f}
   p-value       : {p_value:.6f}
   Conclusion    : {'✅ Différence SIGNIFICATIVE (p<0.05)' if p_value < 0.05 else '❌ Pas de différence'}

1.2. Corrélation Gravité × Durée de séjour
   Coefficient r : {corr:.3f}
   R²            : {corr**2:.3f} ({corr**2*100:.1f}% variance expliquée)
   Interprétation : {'Forte corrélation positive' if corr > 0.7 else 'Corrélation modérée'}

1.3. ANOVA : Durée par service
   F-statistic : {f_stat:.3f}
   p-value     : {p_anova:.6f}
   Conclusion  : {'✅ Différence SIGNIFICATIVE entre services' if p_anova < 0.05 else '❌ Pas de différence'}

1.4. Test de normalité (Shapiro-Wilk)
   Statistique : {stat_shapiro:.3f}
   p-value     : {p_shapiro:.6f}
   Distribution : {'Normale' if p_shapiro > 0.05 else 'Non-normale'}

2. GRAPHIQUES GÉNÉRÉS
{'='*70}

Graph 7 : Heatmap Jour × Mois
   → Permet d'identifier les patterns hebdomadaires et mensuels
   → Justification : Visualisation 2D pour croiser 2 dimensions temporelles

Graph 8 : Boxplot par Mois
   → Montre la distribution et les outliers mensuels
   → Justification : Visualise médiane, quartiles et valeurs extrêmes

Graph 9 : Décomposition Série Temporelle
   → Sépare tendance, saisonnalité et bruit
   → Justification : Analyse structurelle des composantes

Graph 10 : Corrélation Gravité × Durée
   → Quantifie la relation entre gravité et durée
   → Justification : Scatter plot avec régression pour relation linéaire

Graph 11 : Violin Plot par Service
   → Distribution complète de la durée par service
   → Justification : Combine boxplot et densité de probabilité

Graph 12 : Autocorrélation (ACF/PACF)
   → Détecte les dépendances temporelles
   → Justification : Essentiel pour modèles ARIMA/séries temporelles

3. PÉRIODES CRITIQUES IDENTIFIÉES
{'='*70}

Critères : Admissions > Moyenne + 2σ

"""

# Identifier périodes critiques
seuil = df['nb_admissions'].mean() + 2 * df['nb_admissions'].std()
critiques = df[df['nb_admissions'] > seuil]

rapport += f"Seuil critique : {seuil:.1f} admissions/jour\n"
rapport += f"Nombre de jours critiques : {len(critiques)}\n\n"

if len(critiques) > 0:
    rapport += "Périodes critiques détectées :\n"
    for _, row in critiques.iterrows():
        rapport += f"  - {row['date'].strftime('%Y-%m-%d')} : {row['nb_admissions']:.0f} admissions"
        if pd.notna(row.get('event')) and row.get('event') != 'none':
            rapport += f" (Événement: {row['event']})"
        rapport += "\n"

rapport += f"\n{'='*70}\n"
rapport += "Rapport généré le " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n"
rapport += f"{'='*70}\n"

with open("rapport_statistique.txt", "w", encoding='utf-8') as f:
    f.write(rapport)

print("\n" + "="*70)
print("✅ ANALYSE TERMINÉE")
print("="*70)
print("\nFichiers générés:")
print("  - graph7_heatmap_admissions.png")
print("  - graph8_boxplot_mois.png")
print("  - graph9_decomposition.png")
print("  - graph10_correlation.png")
print("  - graph11_distribution_sejour.png")
print("  - graph12_autocorrelation.png")
print("  - rapport_statistique.txt")