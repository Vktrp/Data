import pandas as pd

adm = pd.read_csv("admissions_daily.csv", parse_dates=["date"])
staff = pd.read_csv("staff.csv", parse_dates=["date"])

df = adm.merge(staff, on="date", how="left")

df["ratio_patients_par_infirmier"] = df["nb_admissions"] / df["infirmiers"]
df["ratio_patients_par_medecin"] = df["nb_admissions"] / df["medecins"]
