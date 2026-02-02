import pandas as pd

"""
GÉNÉRATION DES ADMISSIONS QUOTIDIENNES
Agrège les patients par jour
"""

print("🔧 Génération des admissions quotidiennes...")

# Charger les patients
df = pd.read_csv("patients.csv", parse_dates=["date_admission"])

# Agréger par jour
daily = (
    df.groupby(df["date_admission"].dt.date)
    .agg(
        nb_admissions=("patient_id", "count"),
        event=("event", "first")
    )
    .reset_index()
    .rename(columns={"date_admission": "date"})
)

# Formater la date
daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")

# Sauvegarder
daily.to_csv("admissions_daily.csv", index=False)

print(f"✅ admissions_daily.csv généré")
print(f"   {len(daily)} jours de données")
print(f"   Admissions: min={daily['nb_admissions'].min()}, "
      f"max={daily['nb_admissions'].max()}, "
      f"moyenne={daily['nb_admissions'].mean():.1f}")
print(f"   Écart-type: {daily['nb_admissions'].std():.1f}")