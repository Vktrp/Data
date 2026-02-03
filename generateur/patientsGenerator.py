import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]  # .../data

np.random.seed(42)

# ----------------------------
# PARAMÈTRES
# ----------------------------
START = "2014-01-01"
END   = "2024-12-31"

BASE_ADM_PER_DAY = 220
VARIANCE = 45

# Facteurs saisonniers
WINTER_FACTOR = 1.25
SUMMER_FACTOR = 0.85
WEEKEND_FACTOR = 0.90

# Services + répartition
services = ["urgences", "cardiologie", "neurologie", "pediatrie", "reanimation"]
service_p = [0.55, 0.15, 0.12, 0.12, 0.06]

# Gravité
grav_p_normal = [0.22, 0.30, 0.28, 0.15, 0.05]
grav_p_event  = [0.10, 0.22, 0.33, 0.23, 0.12]

# Durée de séjour
LOS_MEAN = {1: 2.0, 2: 3.0, 3: 5.0, 4: 8.0, 5: 12.0}
LOS_MAX = 30

# ----------------------------
# FONCTION ÉVÉNEMENT RÉALISTE
# ----------------------------
def get_event(date):
    """
    Retourne l'événement selon la date RÉELLE
    - Grippe : chaque année janvier-février
    - COVID : seulement 2020-2023 (nov-déc)
    - Canicule : chaque année juillet
    """
    year = date.year
    month = date.month
    day = date.day
    
    # GRIPPE : Tous les ans, janvier-février
    if month in [1, 2]:
        return "grippe", 1.25
    
    # COVID : Seulement 2020-2023, mars OU novembre-décembre
    if year in [2020, 2021, 2022, 2023]:
        # Première vague : mars 2020
        if year == 2020 and month == 3:
            return "covid", 1.40
        # Vagues hivernales : nov-déc 2020-2023
        if month in [11, 12]:
            return "covid", 1.35
    
    # CANICULE : Tous les ans, juillet
    if month == 7 and day >= 15:
        return "canicule", 1.10
    
    # Pas d'événement
    return "none", 1.0

# ----------------------------
# GÉNÉRATION
# ----------------------------
dates = pd.date_range(START, END, freq="D")
rows = []
patient_id = 1

print("=" * 70)
print("🔄 GÉNÉRATION DE PATIENTS AVEC ÉVÉNEMENTS RÉALISTES")
print("=" * 70)

for d in dates:
    month = d.month
    dow = d.dayofweek
    
    # Événement réaliste
    event, event_factor = get_event(d)
    
    # Saisonnalité
    factor = 1.0
    if month in [1, 2, 12]:
        factor *= WINTER_FACTOR
    elif month in [7, 8]:
        factor *= SUMMER_FACTOR
    
    # Weekend
    if dow in [5, 6]:
        factor *= WEEKEND_FACTOR
    
    # Événement
    factor *= event_factor
    
    # Admissions du jour
    lam = BASE_ADM_PER_DAY * factor
    MIN_ADM_PER_DAY = 120
    n_today = int(np.random.poisson(lam))
    n_today = max(MIN_ADM_PER_DAY, n_today)
    
    # Générer patients
    for _ in range(n_today):
        # Gravité
        if event != "none":
            gravite = np.random.choice([1,2,3,4,5], p=grav_p_event)
        else:
            gravite = np.random.choice([1,2,3,4,5], p=grav_p_normal)
        
        # Durée séjour
        mean = LOS_MEAN[gravite]
        shape = 2.0
        scale = mean / shape
        duree = int(np.ceil(np.random.gamma(shape=shape, scale=scale)))
        duree = max(1, min(LOS_MAX, duree))
        
        row = [
            patient_id,
            d,
            np.random.choice(services, p=service_p),
            int(np.random.randint(0, 100)),
            np.random.choice(["M", "F"]),
            int(gravite),
            int(duree),
            1,
            event  # ← Événement réaliste
        ]
        rows.append(row)
        patient_id += 1

df = pd.DataFrame(rows, columns=[
    "patient_id","date_admission","service","age","sexe",
    "gravite","duree_sejour","lits_utilises","event"
])

# Sauvegarder
df.to_csv(BASE_DIR / "patients.csv", index=False)

# Stats
print(f"\n✅ PATIENTS GÉNÉRÉS: {len(df):,}")
print(f"\n📊 STATISTIQUES PAR ÉVÉNEMENT:")
events_count = df['event'].value_counts()
for event, count in events_count.items():
    pct = (count / len(df)) * 100
    print(f"   {event:<12} {count:>8,} patients ({pct:>5.2f}%)")

print(f"\n📅 ÉVÉNEMENTS PAR ANNÉE:")
for year in range(2014, 2025):
    year_data = df[df['date_admission'].dt.year == year]
    events = year_data['event'].value_counts()
    
    covid_count = events.get('covid', 0)
    grippe_count = events.get('grippe', 0)
    canicule_count = events.get('canicule', 0)
    
    print(f"   {year}: COVID={covid_count:>6,} | Grippe={grippe_count:>6,} | Canicule={canicule_count:>6,}")

print("\n" + "=" * 70)
print("✅ patients.csv généré avec événements réalistes")
print("=" * 70)