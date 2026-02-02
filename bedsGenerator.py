import pandas as pd

LITS_TOTAL = 1800

patients = pd.read_csv("patients.csv", parse_dates=["date_admission"])

# Construire les intervalles d'occupation par patient
patients["date_sortie"] = patients["date_admission"] + pd.to_timedelta(patients["duree_sejour"], unit="D")

# Série temporelle jour par jour : lits occupés
dates = pd.date_range(patients["date_admission"].min(), patients["date_sortie"].max(), freq="D")

rows = []
for d in dates:
    occ = ((patients["date_admission"] <= d) & (patients["date_sortie"] > d)).sum()
    dispo = max(0, LITS_TOTAL - occ)
    rows.append([d.strftime("%Y-%m-%d"), LITS_TOTAL, dispo, occ, "all"])

beds = pd.DataFrame(rows, columns=["date","lits_total","lits_disponibles","lits_occupees","service"])
beds.to_csv("beds.csv", index=False)

print("beds.csv généré ✔")
