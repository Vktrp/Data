import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

"""
GÉNÉRATION DES GRAPHIQUES 1-4
Analyse visuelle des données hospitalières
"""

# Configuration du style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)

print("="*70)
print("📊 GÉNÉRATION DES GRAPHIQUES D'ANALYSE")
print("="*70)

# =============================================================================
# GRAPH 1 : Impact des épidémies sur les Admissions
# =============================================================================
print("\n🎨 Graphique 1 : Admissions et événements...")
df_adm = pd.read_csv("admissions_daily.csv", parse_dates=["date"])

plt.figure()
# Courbe principale
sns.lineplot(data=df_adm, x='date', y='nb_admissions', 
             label='Admissions Journalières', color='#1f77b4', linewidth=2)

# Points pour les événements
subset_event = df_adm[df_adm['event'].notna() & (df_adm['event'] != 'none')]

if not subset_event.empty:
    sns.scatterplot(data=subset_event, x='date', y='nb_admissions', 
                   hue='event', s=80, zorder=3, palette='viridis', legend='full')

plt.title("Évolution des Admissions et Impact des Événements (2024)", 
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
# GRAPH 2 : Risque de Saturation des Lits
# =============================================================================
print("🎨 Graphique 2 : Saturation des lits...")
df_beds = pd.read_csv("beds.csv", parse_dates=["date"])

# Calcul du taux d'occupation
df_beds['taux_occupation'] = (df_beds['lits_occupees'] / df_beds['lits_total']) * 100

plt.figure()
sns.lineplot(data=df_beds, x='date', y='taux_occupation', 
            color='#d62728', linewidth=2.5, label="Taux d'occupation")

# Ligne de seuil critique
plt.axhline(y=100, color='black', linestyle='--', linewidth=2, 
           label="Capacité Max (100%)")

# Zone de saturation
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
# GRAPH 3 : Tension sur le Staff
# =============================================================================
print("🎨 Graphique 3 : Tension personnel...")
df_staff = pd.read_csv("staff.csv", parse_dates=["date"])

# Fusion avec admissions
df_rh = pd.merge(df_adm, df_staff, on="date", how="inner")

# Calcul du ratio
df_rh['ratio'] = df_rh['nb_admissions'] / df_rh['infirmiers']

plt.figure()
sns.lineplot(data=df_rh, x='date', y='ratio', color='green', linewidth=2)
plt.axhline(y=df_rh['ratio'].mean(), color='grey', linestyle='--', 
           linewidth=2, label=f"Moyenne: {df_rh['ratio'].mean():.2f}")

plt.title("Pression RH : Nombre de nouveaux patients par Infirmier", 
         fontsize=16, fontweight='bold')
plt.ylabel("Patients / Infirmier", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.legend(fontsize=10)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graph3_tension_staff.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ graph3_tension_staff.png")

# =============================================================================
# GRAPH 4 : Rupture de Stocks (Masques)
# =============================================================================
print("🎨 Graphique 4 : Gestion des stocks...")
df_stocks = pd.read_csv("stocks.csv", parse_dates=["date"])

plt.figure()
# Stock disponible
sns.lineplot(data=df_stocks, x='date', y='masques', 
            label='Stock Disponible', color='orange', linewidth=2.5)
# Seuil d'alerte
sns.lineplot(data=df_stocks, x='date', y='seuil_masques', 
            color='red', linestyle='--', linewidth=2, label='Seuil de commande')

plt.title("Gestion Logistique : Stock de Masques", 
         fontsize=16, fontweight='bold')
plt.ylabel("Quantité", fontsize=12)
plt.xlabel("Date", fontsize=12)

# Zone de commande
plt.fill_between(df_stocks['date'], df_stocks['masques'], df_stocks['seuil_masques'], 
                where=(df_stocks['masques'] < df_stocks['seuil_masques']), 
                color='red', alpha=0.2, label="Zone de commande")

plt.legend(fontsize=10)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graph4_stocks.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ graph4_stocks.png")

print("\n" + "="*70)
print("✅ GRAPHIQUES 1-4 GÉNÉRÉS AVEC SUCCÈS")
print("="*70)
print("\nFichiers créés:")
print("  - graph1_admissions_epidemies.png")
print("  - graph2_saturation_lits.png")
print("  - graph3_tension_staff.png")
print("  - graph4_stocks.png")