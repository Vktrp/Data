import pandas as pd

df = pd.read_csv("patients.csv", parse_dates=["date_admission"])

daily = (
    df.groupby(df["date_admission"].dt.date)
      .agg(
          nb_admissions=("patient_id","count"),
          event=("event","first")
      )
      .reset_index()
      .rename(columns={"date_admission":"date"})
)

daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")

daily.to_csv("admissions_daily.csv", index=False)

print("admissions_daily.csv généré ✔")
