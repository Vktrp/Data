import pandas as pd

"""
GÉNÉRATION DE L'OCCUPATION DES LITS - VERSION CORRIGÉE
Simule l'occupation jour par jour avec calcul correct
"""

print("🔧 Génération de l'occupation des lits (version corrigée)...")

LITS_TOTAL = 1800

# Charger les patients
patients = pd.read_csv("patients.csv", parse_dates=["date_admission"])

print(f"   {len(patients)} patients chargés")

# Calculer la date de sortie pour chaque patient
patients["date_sortie"] = patients["date_admission"] + pd.to_timedelta(
    patients["duree_sejour"], unit="D"
)

# Créer la série temporelle jour par jour
dates = pd.date_range(
    patients["date_admission"].min(),
    patients["date_sortie"].max(),
    freq="D"
)

print(f"   Calcul de l'occupation pour {len(dates)} jours...")

rows = []
for d in dates:
    # Compter combien de patients sont à l'hôpital ce jour
    # Un patient est présent si : date_admission <= d < date_sortie
    occ = ((patients["date_admission"] <= d) & (patients["date_sortie"] > d)).sum()
    dispo = max(0, LITS_TOTAL - occ)
    
    rows.append([
        d.strftime("%Y-%m-%d"),
        LITS_TOTAL,
        dispo,
        occ,
        "all"
    ])

beds = pd.DataFrame(
    rows,
    columns=["date", "lits_total", "lits_disponibles", "lits_occupees", "service"]
)

# Sauvegarder
beds.to_csv("beds.csv", index=False)

print(f"✅ beds.csv généré")
print(f"   {len(beds)} jours de données")
print(f"   Occupation moyenne: {beds['lits_occupees'].mean():.1f} lits "
      f"({(beds['lits_occupees'].mean()/LITS_TOTAL*100):.1f}%)")
print(f"   Occupation max: {beds['lits_occupees'].max()} lits "
      f"({(beds['lits_occupees'].max()/LITS_TOTAL*100):.1f}%)")

# Saturation
saturation = (beds['lits_occupees'] >= LITS_TOTAL).sum()
print(f"   Jours de saturation: {saturation} ({(saturation/len(beds)*100):.1f}%)")

# Vérification
taux_moyen = (beds['lits_occupees'].mean() / LITS_TOTAL) * 100
if taux_moyen < 50:
    print(f"\n⚠️  ATTENTION: Taux d'occupation très bas ({taux_moyen:.1f}%)")
    print(f"   Cela peut indiquer que les durées de séjour sont trop courtes")
    print(f"   Durée moyenne de séjour: {patients['duree_sejour'].mean():.1f} jours")
elif taux_moyen > 90:
    print(f"\n⚠️  ATTENTION: Taux d'occupation très élevé ({taux_moyen:.1f}%)")
    print(f"   L'hôpital est en saturation constante")
else:
    print(f"\n✅ Taux d'occupation réaliste: {taux_moyen:.1f}%")