import pandas as pd
import numpy as np

np.random.seed(42)

# ----------------------------
# PARAMÈTRES À AJUSTER (MVP)
# ----------------------------
START = "2024-01-01"
END   = "2024-12-31"

# Admissions moyennes par jour (calibrage)
BASE_ADM_PER_DAY = 220  # ~300/jour -> avec LOS ~5j => ~1500 lits occupés

# Facteurs saisonniers
WINTER_FACTOR = 1.25   # Jan/Feb/Dec
SUMMER_FACTOR = 0.85   # Jul/Aug
WEEKEND_FACTOR = 0.90  # Sam/Dim

# Événements (impact admissions + gravité + LOS)
events_map = {
    1: "grippe",
    2: "grippe",
    11: "covid",
    12: "covid",
    7: "canicule"
}
EVENT_ADM_MULT = {
    "none": 1.00,
    "grippe": 1.25,
    "covid": 1.35,
    "canicule": 1.10
}

# Services + répartition (somme = 1)
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
    month = d.month
    dow = d.dayofweek  # 0=lun ... 5=sam 6=dim

    event = events_map.get(month, "none")

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

    # générer n_today patients
    for _ in range(n_today):
        # gravité
        if event != "none":
            gravite = np.random.choice([1,2,3,4,5], p=grav_p_event)
        else:
            gravite = np.random.choice([1,2,3,4,5], p=grav_p_normal)

        # durée de séjour (gamma = asymétrique, plus réaliste qu'une normale)
        mean = LOS_MEAN[gravite]
        shape = 2.0
        scale = mean / shape
        duree = int(np.ceil(np.random.gamma(shape=shape, scale=scale)))
        duree = max(1, min(LOS_MAX, duree))

        row = [
            patient_id,
            d.strftime("%Y-%m-%d"),
            np.random.choice(services, p=service_p),
            int(np.random.randint(0, 100)),
            np.random.choice(["M", "F"]),
            int(gravite),
            int(duree),
            1,
            event
        ]
        rows.append(row)
        patient_id += 1

df = pd.DataFrame(rows, columns=[
    "patient_id","date_admission","service","age","sexe",
    "gravite","duree_sejour","lits_utilises","event"
])

df.to_csv("patients.csv", index=False)
print(f"patients.csv généré ✔  (N={len(df)})")
