import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import pickle
import warnings
warnings.filterwarnings('ignore')

"""
GÉNÉRATION COMPLÈTE DE TOUS LES GRAPHIQUES
Graphiques conservés : 1, 2, 7, 10, 13, 14 + A, B, C
Graphiques supprimés : 3, 4, 8, 9, 11, 12
"""

print("="*80)
print("📊 GÉNÉRATION COMPLÈTE DE TOUS LES GRAPHIQUES PERTINENTS")
print("="*80)

# Configuration
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.figsize': (14, 7),
    'font.size': 11
})

# =============================================================================
# CHARGEMENT DONNÉES
# =============================================================================

print("\n📂 Chargement des données...")
try:
    df_adm = pd.read_csv("admissions_daily.csv", parse_dates=["date"])
    df_beds = pd.read_csv("beds.csv", parse_dates=["date"])
    df_patients = pd.read_csv("patients.csv", parse_dates=["date_admission"])
    print("✅ Données chargées")
except FileNotFoundError as e:
    print(f"❌ Erreur : {e}")
    exit(1)

# =============================================================================
# GRAPH 1 : ADMISSIONS + ÉPIDÉMIES
# =============================================================================

print("\n📊 Graph 1 : Admissions et événements...")

plt.figure(figsize=(14, 7))
sns.lineplot(data=df_adm, x='date', y='nb_admissions', 
             label='Admissions Journalières', color='#1f77b4', linewidth=2)

subset_event = df_adm[df_adm['event'].notna() & (df_adm['event'] != 'none')]
if not subset_event.empty:
    sns.scatterplot(data=subset_event, x='date', y='nb_admissions', 
                   hue='event', s=80, zorder=3, palette='viridis', legend='full')

plt.title("Évolution des Admissions et Impact des Événements", 
         fontsize=16, fontweight='bold')
plt.ylabel("Nombre d'admissions", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.legend(title="Événement", fontsize=10)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graph1_admissions_epidemies.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ graph1_admissions_epidemies.png")

# =============================================================================
# GRAPH 2 : SATURATION DES LITS
# =============================================================================

print("📊 Graph 2 : Saturation des lits...")

df_beds['taux_occupation'] = (df_beds['lits_occupees'] / df_beds['lits_total']) * 100

plt.figure(figsize=(14, 7))
sns.lineplot(data=df_beds, x='date', y='taux_occupation', 
            color='#d62728', linewidth=2.5, label="Taux d'occupation")

plt.axhline(y=100, color='black', linestyle='--', linewidth=2, 
           label="Capacité Max (100%)")

plt.fill_between(df_beds['date'], df_beds['taux_occupation'], 100, 
                where=(df_beds['taux_occupation'] >= 100), 
                color='red', alpha=0.3, label="Saturation")

plt.title("Tension Hospitalière : Taux d'Occupation des Lits", 
         fontsize=16, fontweight='bold')
plt.ylabel("Occupation (%)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.ylim(0, 110)
plt.legend(fontsize=10)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graph2_saturation_lits.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ graph2_saturation_lits.png")

# =============================================================================
# GRAPH 7 : HEATMAP JOUR × MOIS
# =============================================================================

print("📊 Graph 7 : Heatmap jour × mois...")

df_adm['jour_semaine'] = df_adm['date'].dt.day_name()
df_adm['mois'] = df_adm['date'].dt.month

pivot = df_adm.pivot_table(
    values='nb_admissions',
    index='jour_semaine',
    columns='mois',
    aggfunc='mean'
)

jours_ordre = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot = pivot.reindex(jours_ordre)

plt.figure(figsize=(12, 6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', 
            cbar_kws={'label': 'Admissions moyennes'})
plt.title('Heatmap : Admissions moyennes par Jour × Mois', 
         fontsize=14, fontweight='bold')
plt.ylabel('Jour de la semaine', fontsize=11)
plt.xlabel('Mois', fontsize=11)
plt.tight_layout()
plt.savefig("graph7_heatmap_admissions.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ graph7_heatmap_admissions.png")

# =============================================================================
# GRAPH 10 : CORRÉLATION GRAVITÉ × DURÉE
# =============================================================================

print("📊 Graph 10 : Corrélation gravité × durée...")

corr = df_patients['gravite'].corr(df_patients['duree_sejour'])

plt.figure(figsize=(10, 6))
plt.scatter(df_patients['gravite'], df_patients['duree_sejour'], 
          alpha=0.3, s=20, color='#3498DB')

z = np.polyfit(df_patients['gravite'], df_patients['duree_sejour'], 1)
p = np.poly1d(z)
x_line = np.linspace(1, 5, 100)
plt.plot(x_line, p(x_line), "r--", linewidth=2, 
        label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

plt.xlabel('Gravité (1-5)', fontsize=11)
plt.ylabel('Durée de séjour (jours)', fontsize=11)
plt.title(f'Corrélation Gravité × Durée de séjour (r = {corr:.3f}, R² = {corr**2:.3f})', 
         fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graph10_correlation.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ graph10_correlation.png")


# =============================================================================
# RÉCAPITULATIF
# =============================================================================

print("\n" + "="*80)
print("✅ GÉNÉRATION TERMINÉE")
print("="*80)

print("\n🎯 GRAPHIQUES OPÉRATIONNELS :")
print("   ✅ graph1_admissions_epidemies.png    - Impact événements")
print("   ✅ graph2_saturation_lits.png          - Tension hospitalière")

print("\n📊 GRAPHIQUES STATISTIQUES :")
print("   ✅ graph7_heatmap_admissions.png       - Patterns jour×mois")
print("   ✅ graph10_correlation.png             - Gravité×Durée")
