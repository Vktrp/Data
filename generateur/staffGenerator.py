import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # .../data
np.random.seed(42)

# Lire beds.csv pour caler le staff sur l'occupation réelle
beds = pd.read_csv(BASE_DIR / "beds.csv", parse_dates=["date"]).sort_values("date")

data = []

# Ratios cibles (présents par lits occupés) - MVP cohérent
# Ajuste si besoin : plus le ratio est bas, plus la tension staff monte.
INF_PER_OCC_BED = 1 / 6.0     # ~1 infirmier pour 6 lits occupés
MED_PER_OCC_BED = 1 / 25.0    # ~1 médecin pour 25 lits occupés
AIDE_PER_OCC_BED = 1 / 10.0   # ~1 aide-soignant pour 10 lits occupés

# Variations
WINTER_FACTOR = 1.08   # renfort hiver
SUMMER_FACTOR = 0.90   # sous-effectif été
WEEKEND_FACTOR = 0.92  # moins de présents week-end

for _, r in beds.iterrows():
    d = r["date"]
    occ = int(r["lits_occupees"])
    month = int(d.month)
    dow = int(d.dayofweek)  # 0=lun ... 5=sam 6=dim

    factor = 1.0
    if month in [1, 2, 12]:
        factor *= WINTER_FACTOR
    elif month in [7, 8]:
        factor *= SUMMER_FACTOR

    if dow in [5, 6]:
        factor *= WEEKEND_FACTOR

    # Base calculée sur l'occupation
    med = occ * MED_PER_OCC_BED * factor
    inf = occ * INF_PER_OCC_BED * factor
    aide = occ * AIDE_PER_OCC_BED * factor

    # Bruit réaliste (évite série trop "parfaite")
    med = int(round(med + np.random.normal(0, 2)))
    inf = int(round(inf + np.random.normal(0, 6)))
    aide = int(round(aide + np.random.normal(0, 4)))

    # Planchers proportionnels : évite l'effet "collé au minimum"
    med = max(0, med)
    inf = max(0, inf)
    aide = max(0, aide)

    # Planchers relatifs (si très faible activité, on garde une équipe de base)
    med = max(med, int(round(occ / 60)))     # ex: 337/60 ≈ 6 -> donc pas dominant ici
    inf = max(inf, int(round(occ / 12)))     # ex: 337/12 ≈ 28
    aide = max(aide, int(round(occ / 18)))   # ex: 337/18 ≈ 19

    data.append([d.strftime("%Y-%m-%d"), med, inf, aide])

staff = pd.DataFrame(data, columns=["date", "medecins", "infirmiers", "aides_soignants"])
staff.to_csv(BASE_DIR / "staff.csv", index=False)

print("staff.csv généré ✔")
