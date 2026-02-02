import pandas as pd
import numpy as np

"""
GÉNÉRATEUR DE PATIENTS - VERSION AMÉLIORÉE
Génère des données réalistes avec patterns prévisibles
"""

np.random.seed(42)

# =============================================================================
# PARAMÈTRES
# =============================================================================

BASELINE_ADMISSIONS = 40  # Niveau de base
BRUIT_ALEATOIRE = 0.15    # 15% de bruit réaliste
EFFET_SAISON = 0.25       # 25% variation saisonnière
EFFET_WEEKEND = -0.20     # -20% le weekend
EFFET_LUNDI = 0.15        # +15% le lundi
EFFET_EPIDEMIE = 0.40     # +40% pendant épidémies

# =============================================================================
# GÉNÉRATION DES ADMISSIONS PAR JOUR
# =============================================================================

print("🔧 Génération des patients...")

dates = pd.date_range("2024-01-01", "2024-12-31")
services = ["urgences", "cardiologie", "neurologie", "pediatrie", "reanimation"]

# D'abord, calculer le nombre d'admissions par jour
admissions_par_jour = []

for date in dates:
    # 1. BASE
    admissions = BASELINE_ADMISSIONS
    
    # 2. EFFET SAISONNIER (sinusoïde)
    jour_annee = date.dayofyear
    effet_hiver = np.sin(2 * np.pi * (jour_annee - 15) / 365)
    admissions *= (1 + EFFET_SAISON * effet_hiver)
    
    # 3. EFFET JOUR DE LA SEMAINE
    jour_semaine = date.dayofweek
    if jour_semaine == 0:  # Lundi
        admissions *= (1 + EFFET_LUNDI)
    elif jour_semaine >= 5:  # Weekend
        admissions *= (1 + EFFET_WEEKEND)
    
    # 4. ÉVÉNEMENTS
    mois = date.month
    event = "none"
    
    if mois in [1, 2]:  # Grippe
        event = "grippe"
        intensite = np.sin(np.pi * (date.day / 60))
        admissions *= (1 + EFFET_EPIDEMIE * intensite)
    elif mois == 7:  # Canicule
        event = "canicule"
        if 10 <= date.day <= 25:
            admissions *= (1 + EFFET_EPIDEMIE * 0.7)
    elif mois in [11, 12]:  # COVID
        event = "covid"
        intensite = (date.day / 30) * 0.8
        admissions *= (1 + EFFET_EPIDEMIE * intensite)
    
    # 5. BRUIT ALÉATOIRE
    bruit = np.random.normal(1, BRUIT_ALEATOIRE)
    admissions *= bruit
    
    # 6. ARRONDIR
    admissions = int(max(15, round(admissions)))
    
    admissions_par_jour.append({
        'date': date,
        'nb_admissions': admissions,
        'event': event
    })

df_admissions_jour = pd.DataFrame(admissions_par_jour)

# =============================================================================
# GÉNÉRATION DES PATIENTS INDIVIDUELS
# =============================================================================

patients_data = []
patient_id = 1

for _, row in df_admissions_jour.iterrows():
    date = row['date']
    nb_adm = row['nb_admissions']
    event = row['event']
    
    # Générer exactement nb_adm patients ce jour
    for _ in range(nb_adm):
        # Gravité selon événement
        if event in ['grippe', 'covid']:
            gravite = np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3])
        elif event == 'canicule':
            gravite = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])
        else:
            gravite = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.25, 0.30, 0.20, 0.10])
        
        # Durée corrélée à gravité
        duree_base = gravite * 3
        duree = int(max(1, np.random.normal(duree_base, 1.5)))
        
        patients_data.append({
            'patient_id': patient_id,
            'date_admission': date,
            'service': np.random.choice(services),
            'age': np.random.randint(0, 100),
            'sexe': np.random.choice(['M', 'F']),
            'gravite': gravite,
            'duree_sejour': duree,
            'lits_utilises': 1,
            'event': event
        })
        patient_id += 1

df = pd.DataFrame(patients_data)
df.to_csv("patients.csv", index=False)

print(f"✅ patients.csv généré ({len(df)} patients)")
print(f"   Période: 2024-01-01 → 2024-12-31")
print(f"   Services: {', '.join(services)}")
print(f"   Admissions/jour: min={df_admissions_jour['nb_admissions'].min()}, "
      f"max={df_admissions_jour['nb_admissions'].max()}, "
      f"moy={df_admissions_jour['nb_admissions'].mean():.1f}")