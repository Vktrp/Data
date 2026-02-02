import pandas as pd
import numpy as np

"""
GÉNÉRATION DES DONNÉES DE PERSONNEL (STAFF)
Médecins et infirmiers avec variations réalistes
"""

print("🔧 Génération des données de personnel...")

dates = pd.date_range("2024-01-01", "2024-12-31")
staff_data = []

for date in dates:
    jour_semaine = date.dayofweek
    mois = date.month
    
    # Configuration de base
    if jour_semaine >= 5:  # Weekend
        medecins = 25
        infirmiers = 60
    else:  # Semaine
        medecins = 30
        infirmiers = 75
    
    # Augmentation en période hivernale (grippe, covid)
    if mois in [1, 2, 11, 12]:
        medecins += 5
        infirmiers += 10
    
    staff_data.append({
        'date': date.strftime("%Y-%m-%d"),
        'medecins': medecins,
        'infirmiers': infirmiers
    })

df_staff = pd.DataFrame(staff_data)
df_staff.to_csv("staff.csv", index=False)

print(f"✅ staff.csv généré")
print(f"   {len(df_staff)} jours de données")
print(f"   Médecins: {df_staff['medecins'].min()}-{df_staff['medecins'].max()}")
print(f"   Infirmiers: {df_staff['infirmiers'].min()}-{df_staff['infirmiers'].max()}")