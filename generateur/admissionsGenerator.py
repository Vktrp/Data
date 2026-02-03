import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # .../data

df = pd.read_csv(BASE_DIR / "patients.csv", parse_dates=["date_admission"])

# colonne de regroupement explicite
df["date"] = df["date_admission"].dt.floor("D")

# 1) Admissions par jour et par service
admissions = (
    df.groupby(["date", "service"], as_index=False)
      .agg(nb_admissions=("patient_id", "count"))
)

# 2) Total tous services ("all")
admissions_total = (
    df.groupby("date", as_index=False)
      .agg(nb_admissions=("patient_id", "count"))
)
admissions_total["service"] = "all"

admissions = pd.concat([admissions, admissions_total], ignore_index=True)

# 3) Format + tri + export
admissions["date"] = admissions["date"].dt.strftime("%Y-%m-%d")
admissions = admissions.sort_values(["date", "service"])

admissions.to_csv(BASE_DIR / "admissions.csv", index=False)
print("admissions.csv généré ✔")
