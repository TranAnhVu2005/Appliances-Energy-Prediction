import pandas as pd
df = pd.read_csv(
    "D:/MayHocUngDung_TeamWork/Appliances-Energy-Prediction/appliances+energy+prediction/energydata_complete.csv", sep=",")
print(df.head())
print(df.info())
print(df.describe())
