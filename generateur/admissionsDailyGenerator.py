import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # .../data

df = pd.read_csv(BASE_DIR / "patients.csv", parse_dates=["date_admission"])

# Créer une colonne de regroupement explicite (évite FutureWarning + KeyError)
df["date"] = df["date_admission"].dt.floor("D")

daily = (
    df.groupby("date", as_index=False)
      .agg(
          nb_admissions=("patient_id", "count"),
          event=("event", "first")
      )
)

daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

daily.to_csv(BASE_DIR / "admissions_daily.csv", index=False)
print("admissions_daily.csv généré ✔")
