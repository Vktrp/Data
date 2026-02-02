import pandas as pd
import numpy as np

np.random.seed(42)

adm = pd.read_csv("admissions_daily.csv", parse_dates=["date"])
adm = adm.sort_values("date").reset_index(drop=True)

# Seuils (alertes)
SEUILS = {
    "masques": 25000,
    "blouses": 8000,
    "respirateurs": 120,
    "tests": 7000,
    "gel": 4000
}

# Stocks initiaux (arbitraires mais cohérents)
stock = {
    "masques": 60000,
    "blouses": 20000,
    "respirateurs": 220,
    "tests": 15000,
    "gel": 9000
}

# Consommation de base par admission (moyennes)
BASE_USE = {
    "masques": 2.0,
    "blouses": 0.6,
    "respirateurs": 0.02,  # faible, réservé aux cas graves
    "tests": 0.7,
    "gel": 0.2
}

# Multiplicateurs par événement
EVENT_MULT = {
    "none": {"masques": 1.0, "tests": 1.0, "gel": 1.0, "blouses": 1.0, "respirateurs": 1.0},
    "grippe": {"masques": 1.4, "tests": 1.3, "gel": 1.2, "blouses": 1.1, "respirateurs": 1.0},
    "covid": {"masques": 2.0, "tests": 1.8, "gel": 1.5, "blouses": 1.2, "respirateurs": 1.1},
    "canicule": {"masques": 1.0, "tests": 1.0, "gel": 1.1, "blouses": 1.0, "respirateurs": 1.05}
}

# Réassort si sous seuil (quantités fixes)
RESTOCK_QTY = {
    "masques": 50000,
    "blouses": 15000,
    "respirateurs": 60,
    "tests": 20000,
    "gel": 8000
}

rows = []

for _, r in adm.iterrows():
    date = r["date"]
    nb = int(r["nb_admissions"])
    event = r["event"] if "event" in adm.columns else "none"
    if pd.isna(event):
        event = "none"

    mult = EVENT_MULT.get(event, EVENT_MULT["none"])

    # consommation du jour (avec bruit)
    use_masques = int(nb * BASE_USE["masques"] * mult["masques"] + np.random.normal(0, nb * 0.05))
    use_blouses = int(nb * BASE_USE["blouses"] * mult["blouses"] + np.random.normal(0, nb * 0.02))
    use_tests = int(nb * BASE_USE["tests"] * mult["tests"] + np.random.normal(0, nb * 0.03))
    use_gel = int(nb * BASE_USE["gel"] * mult["gel"] + np.random.normal(0, nb * 0.01))

    # respirateurs : dépend aussi de la gravité moyenne estimée (proxy via event)
    base_resp = nb * BASE_USE["respirateurs"] * mult["respirateurs"]
    use_resp = int(max(0, base_resp + np.random.normal(0, 1)))

    # éviter négatifs
    use_masques = max(0, use_masques)
    use_blouses = max(0, use_blouses)
    use_tests = max(0, use_tests)
    use_gel = max(0, use_gel)
    use_resp = max(0, use_resp)

    # décrémenter stocks
    stock["masques"] -= use_masques
    stock["blouses"] -= use_blouses
    stock["tests"] -= use_tests
    stock["gel"] -= use_gel
    stock["respirateurs"] -= use_resp

    # pas de stock négatif
    for k in stock:
        stock[k] = max(0, stock[k])

    # réassort si sous seuil
    for k, seuil in SEUILS.items():
        if stock[k] < seuil:
            stock[k] += RESTOCK_QTY[k]

    rows.append([
        date.strftime("%Y-%m-%d"),
        stock["masques"],
        stock["blouses"],
        stock["respirateurs"],
        stock["tests"],
        stock["gel"],
        SEUILS["masques"],
        SEUILS["blouses"],
        SEUILS["respirateurs"],
        SEUILS["tests"],
        SEUILS["gel"]
    ])

stocks = pd.DataFrame(rows, columns=[
    "date","masques","blouses","respirateurs","tests","gel",
    "seuil_masques","seuil_blouses","seuil_respirateurs","seuil_tests","seuil_gel"
])

stocks.to_csv("stocks.csv", index=False)
print("stocks.csv généré ✔")
