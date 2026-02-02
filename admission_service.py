import pandas as pd

"""
GÉNÉRATION DES ADMISSIONS PAR SERVICE
Agrège par jour ET par service
"""

print("🔧 Génération des admissions par service...")

# Charger les patients
df = pd.read_csv("patients.csv", parse_dates=["date_admission"])

# Agréger par jour et par service
admissions = (
    df.groupby([df["date_admission"].dt.date, "service"])
    .size()
    .reset_index(name="nb_admissions")
    .rename(columns={"date_admission": "date"})
)

# Ajouter une ligne "all" = total tous services
admissions_total = (
    df.groupby(df["date_admission"].dt.date)
    .size()
    .reset_index(name="nb_admissions")
    .rename(columns={"date_admission": "date"})
)
admissions_total["service"] = "all"

# Combiner
admissions = pd.concat([admissions, admissions_total], ignore_index=True)

# Formater et trier
admissions["date"] = pd.to_datetime(admissions["date"]).dt.strftime("%Y-%m-%d")
admissions = admissions.sort_values(["date", "service"])

# Sauvegarder
admissions.to_csv("admissions.csv", index=False)

print(f"✅ admissions.csv généré")
print(f"   {len(admissions)} lignes (jours × services)")
print(f"   Services: {', '.join(admissions[admissions['service'] != 'all']['service'].unique())}")