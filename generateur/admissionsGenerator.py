import pandas as pd

# 1) Lire patients.csv
df = pd.read_csv("patients.csv", parse_dates=["date_admission"])

# 2) Agréger : admissions par jour et par service
admissions = (
    df.groupby([df["date_admission"].dt.date, "service"])
      .size()
      .reset_index(name="nb_admissions")
      .rename(columns={"date_admission": "date"})
)

# 3) (Optionnel mais utile) Ajouter une ligne "all" = total tous services
admissions_total = (
    df.groupby(df["date_admission"].dt.date)
      .size()
      .reset_index(name="nb_admissions")
      .rename(columns={"date_admission": "date"})
)
admissions_total["service"] = "all"

admissions = pd.concat([admissions, admissions_total], ignore_index=True)

# 4) Trier et exporter
admissions["date"] = pd.to_datetime(admissions["date"]).dt.strftime("%Y-%m-%d")
admissions = admissions.sort_values(["date", "service"])

admissions.to_csv("admissions.csv", index=False)

print("admissions.csv généré ✔")
