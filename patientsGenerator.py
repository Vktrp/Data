import pandas as pd
import numpy as np

np.random.seed(42)

N = 15000  # nombre de patients simulés

dates = pd.date_range("2024-01-01", "2024-12-31")

services = ["urgences", "cardiologie", "neurologie", "pediatrie", "reanimation"]
events_map = {
    1: "grippe",
    2: "grippe",
    11: "covid",
    12: "covid",
    7: "canicule"
}

data = []

for i in range(N):

    date = pd.Timestamp(np.random.choice(dates))
    month = date.month
    
    event = events_map.get(month, "none")

    # pics d'admissions en hiver
    if event != "none":
        gravite = np.random.choice([3,4,5], p=[0.3,0.4,0.3])
    else:
        gravite = np.random.choice([1,2,3,4,5], p=[0.2,0.3,0.3,0.15,0.05])

    # durée séjour corrélée gravité
    duree = int(np.random.normal(gravite*2, 1.5))
    duree = max(1, duree)

    row = [
        i+1,
        date,
        np.random.choice(services),
        np.random.randint(0, 100),
        np.random.choice(["M","F"]),
        gravite,
        duree,
        1,
        event
    ]

    data.append(row)

df = pd.DataFrame(data, columns=[
    "patient_id","date_admission","service","age","sexe",
    "gravite","duree_sejour","lits_utilises","event"
])

df.to_csv("patients.csv", index=False)

print("patients.csv généré ✔")
