import pandas as pd

s = pd.read_csv("stocks.csv")
print(s.head())
print(s.describe())
