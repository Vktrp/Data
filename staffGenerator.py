import pandas as pd
import numpy as np

dates = pd.date_range("2024-01-01", "2024-12-31")

data = []

for d in dates:
    
    month = d.month
    
    base_med = 45
    base_inf = 110
    base_aide = 70

    # vacances été → moins staff
    if month in [7,8]:
        factor = 0.85
    # hiver → renfort
    elif month in [1,2,12]:
        factor = 1.1
    else:
        factor = 1

    med = int(base_med * factor + np.random.normal(0,2))
    inf = int(base_inf * factor + np.random.normal(0,5))
    aide = int(base_aide * factor + np.random.normal(0,3))

    data.append([d.strftime("%Y-%m-%d"), med, inf, aide])

staff = pd.DataFrame(data, columns=[
    "date","medecins","infirmiers","aides_soignants"
])

staff.to_csv("staff.csv", index=False)

print("staff.csv généré ✔")
