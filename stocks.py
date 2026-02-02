import pandas as pd
import numpy as np

"""
GÉNÉRATION DES DONNÉES DE STOCKS (MASQUES) - CORRIGÉ
Pattern réaliste sans dents de scie
"""

print("🔧 Génération des données de stocks...")

# Charger les admissions
df_adm = pd.read_csv("admissions_daily.csv", parse_dates=["date"])

dates = pd.date_range("2024-01-01", "2024-12-31")
stocks_data = []

stock_masques = 10000
seuil_masques = 3000
np.random.seed(42)

jours_depuis_dernier_reappro = 0

for i, date in enumerate(dates):
    # Consommation basée sur admissions avec variation
    adm_jour = df_adm.loc[df_adm['date'] == date.strftime("%Y-%m-%d"), 'nb_admissions']
    
    if not adm_jour.empty:
        base = int(adm_jour.values[0] * 15)
        variation = np.random.uniform(0.85, 1.15)  # ±15%
        consommation = int(base * variation)
    else:
        consommation = 600
    
    # Consommer
    stock_masques -= consommation
    
    # Réapprovisionnement plus réaliste
    jours_depuis_dernier_reappro += 1
    
    # On commande seulement si :
    # 1. Stock < seuil
    # 2. Ça fait au moins 5 jours depuis la dernière commande
    if stock_masques < seuil_masques and jours_depuis_dernier_reappro >= 5:
        # Quantité variable selon l'urgence
        if stock_masques < 2000:  # Très urgent
            reappro = np.random.randint(8000, 10000)
        else:  # Normal
            reappro = np.random.randint(6000, 8000)
        
        stock_masques += reappro
        jours_depuis_dernier_reappro = 0  # Reset le compteur
    
    # Éviter stock négatif (sécurité)
    stock_masques = max(2000, stock_masques)
    
    stocks_data.append({
        'date': date.strftime("%Y-%m-%d"),
        'masques': int(stock_masques),
        'seuil_masques': seuil_masques
    })

df_stocks = pd.DataFrame(stocks_data)
df_stocks.to_csv("stocks.csv", index=False)

print(f"✅ stocks.csv généré ({len(df_stocks)} jours)")
print(f"   Stock moyen: {df_stocks['masques'].mean():.0f}")
print(f"   Min/Max: {df_stocks['masques'].min()}/{df_stocks['masques'].max()}")

# Compter les réapprovisionnements
reappro = (df_stocks['masques'].diff() > 4000).sum()
print(f"   Réapprovisionnements: {reappro} fois")