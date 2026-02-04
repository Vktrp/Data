import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # .../data

LITS_TOTAL = 1800

patients = pd.read_csv(BASE_DIR / "patients.csv", parse_dates=["date_admission"])

# On borne la série à la dernière date d'admission (évite d'aller en 2025 à cause des séjours)
END_DATE = patients["date_admission"].max().floor("D")

# Construire les intervalles d'occupation par patient
patients["date_sortie"] = patients["date_admission"] + pd.to_timedelta(patients["duree_sejour"], unit="D")

# Série temporelle jour par jour : bornée à END_DATE
dates = pd.date_range(
    patients["date_admission"].min(),
    min(patients["date_sortie"].max(), END_DATE),
    freq="D"
)

rows = []
for d in dates:
    occ = ((patients["date_admission"] <= d) & (patients["date_sortie"] > d)).sum()

    dispo = max(0, LITS_TOTAL - occ)
    surcapacite = max(0, occ - LITS_TOTAL)

    taux_occupation = occ / LITS_TOTAL
    taux_occupation_cap = min(taux_occupation, 1.0)  # pour graphe % lisible

    rows.append([
        d.strftime("%Y-%m-%d"),
        LITS_TOTAL,
        dispo,
        occ,
        taux_occupation,
        taux_occupation_cap,
        surcapacite,
        "all"
    ])

beds = pd.DataFrame(
    rows,
    columns=[
        "date",
        "lits_total",
        "lits_disponibles",
        "lits_occupees",
        "taux_occupation",
        "taux_occupation_cap",
        "surcapacite",
        "service"
    ],
)

def risk_level(dispo, taux, surcap):
    # Si dépassement de capacité, on expose explicitement la crise
    if surcap > 0:
        return "suroccupation"
    if dispo < 100 or taux >= 0.95:
        return "critique"
    if dispo < 300 or taux >= 0.90:
        return "tension"
    return "normal"

beds["niveau_risque"] = beds.apply(
    lambda r: risk_level(r["lits_disponibles"], r["taux_occupation"], r["surcapacite"]),
    axis=1
)

# Arrondis pour lisibilité
beds["taux_occupation"] = beds["taux_occupation"].round(3)
beds["taux_occupation_cap"] = beds["taux_occupation_cap"].round(3)

beds = beds.sort_values("date")
beds.to_csv(BASE_DIR / "beds.csv", index=False)
print("beds.csv généré ✔")
