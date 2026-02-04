import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # .../data
np.random.seed(42)

# ----------------------------
# PARAMÈTRES (MVP + multi-années)
# ----------------------------
START = "2014-01-01"
END   = "2024-12-31"

# Admissions moyennes par jour (calibrage global)
BASE_ADM_PER_DAY = 220

# Facteurs saisonniers
WINTER_FACTOR  = 1.25   # Jan/Feb/Dec
SUMMER_FACTOR  = 0.85   # Jul/Aug
WEEKEND_FACTOR = 0.90   # Sam/Dim

# Minimum d'admissions/jour (évite jours trop bas) : proportionnel à la base
MIN_ADM_PER_DAY = int(BASE_ADM_PER_DAY * 0.40)  # ex: 88 si base=220

# ----------------------------
# Événements (historiquement cohérents)
# ----------------------------
def get_event(date: pd.Timestamp) -> str:
    y = int(date.year)
    m = int(date.month)

    # Canicule en été (toutes années)
    if m == 7:
        return "canicule"

    # Grippe en hiver (toutes années)
    if m in [1, 2]:
        return "grippe"

    # COVID : à partir de 2020 seulement
    # - vagues plus fortes 2020-2021 (plusieurs mois)
    if y in [2020, 2021] and m in [3, 4, 10, 11, 12]:
        return "covid"
    # - vagues saisonnières ensuite (fin d'année)
    if y >= 2022 and m in [11, 12]:
        return "covid"

    return "none"

# Multiplicateurs d'admissions par événement
EVENT_ADM_MULT = {
    "none": 1.00,
    "grippe": 1.25,
    "covid": 1.35,
    "canicule": 1.10,
}

# ----------------------------
# Services + répartition (somme = 1)
# ----------------------------
services = ["urgences", "cardiologie", "neurologie", "pediatrie", "reanimation"]
service_p = [0.55, 0.15, 0.12, 0.12, 0.06]

# Gravité (1-5) selon contexte
grav_p_normal = [0.22, 0.30, 0.28, 0.15, 0.05]
grav_p_event  = [0.10, 0.22, 0.33, 0.23, 0.12]  # + grave en période d'event

# Durée de séjour par gravité (moyennes approx en jours)
LOS_MEAN = {1: 2.0, 2: 3.0, 3: 5.0, 4: 8.0, 5: 12.0}
LOS_MAX = 30  # plafond simple

# ----------------------------
# GÉNÉRATION JOUR PAR JOUR
# ----------------------------
dates = pd.date_range(START, END, freq="D")
rows = []
patient_id = 1

for d in dates:
    month = int(d.month)
    dow = int(d.dayofweek)  # 0=lun ... 5=sam 6=dim

    event = get_event(d)

    # saisonnalité
    factor = 1.0
    if month in [1, 2, 12]:
        factor *= WINTER_FACTOR
    elif month in [7, 8]:
        factor *= SUMMER_FACTOR

    # week-end
    if dow in [5, 6]:
        factor *= WEEKEND_FACTOR

    # événement
    factor *= EVENT_ADM_MULT.get(event, 1.0)

    # admissions du jour (Poisson = réaliste pour comptages)
    lam = BASE_ADM_PER_DAY * factor
    n_today = int(np.random.poisson(lam))
    n_today = max(MIN_ADM_PER_DAY, n_today)

    for _ in range(n_today):
        # gravité
        if event != "none":
            gravite = np.random.choice([1, 2, 3, 4, 5], p=grav_p_event)
        else:
            gravite = np.random.choice([1, 2, 3, 4, 5], p=grav_p_normal)

        # durée de séjour (gamma = asymétrique, plus réaliste qu'une normale)
        mean = float(LOS_MEAN[int(gravite)])
        shape = 2.0
        scale = mean / shape
        duree = int(np.ceil(np.random.gamma(shape=shape, scale=scale)))
        duree = max(1, min(LOS_MAX, duree))

        rows.append([
            patient_id,
            d,  # datetime64 (mieux que string pour parsings futurs)
            np.random.choice(services, p=service_p),
            int(np.random.randint(0, 100)),
            np.random.choice(["M", "F"]),
            int(gravite),
            int(duree),
            1,
            event
        ])
        patient_id += 1

df = pd.DataFrame(rows, columns=[
    "patient_id", "date_admission", "service", "age", "sexe",
    "gravite", "duree_sejour", "lits_utilises", "event"
])

out_path = BASE_DIR / "patients.csv"
df.to_csv(out_path, index=False)

print(f"patients.csv généré ✔  (N={len(df)}) -> {out_path}")
print(f"Période: {df['date_admission'].min()} -> {df['date_admission'].max()}")
print("Admissions/jour (moyenne):", df.groupby(df["date_admission"].dt.floor("D")).size().mean())
print("Durée séjour (moyenne):", df["duree_sejour"].mean())
