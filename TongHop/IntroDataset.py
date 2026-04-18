import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load dữ liệu
df = pd.read_csv(
    "D:/All/Information Technology  - CTU/CurrentSemester/CT294 - Applied machine learning/MayHocUngDung_TeamWork/Appliances-Energy-Prediction/appliances+energy+prediction/energydata_complete.csv", sep=",")
df['date'] = pd.to_datetime(df['date'])
df['Hour'] = df['date'].dt.hour
df['DayOfWeek'] = df['date'].dt.dayofweek


# Tiền xử lý
X = df.drop(["Appliances", "date", "rv1",
             "rv2"], axis=1)


Y = df["Appliances"]
# Train/Test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=False
)

# Cho train
sample = X_train.iloc[:, :10].sample(10, random_state=42)
samplelabel = Y_train.loc[sample.index]

# scale hoặc không, comment hoặc gỡ comment dòng này
scaler = MinMaxScaler()
sample = pd.DataFrame(
    scaler.fit_transform(sample),
    columns=sample.columns,
    index=sample.index
)

sample.insert(0, "Appliances", samplelabel)
# print(sample)

# Cho test
sampleTest = X_test.iloc[:, :10].sample(2, random_state=42)
# scale hoặc không, comment hoặc gỡ comment dòng này
sampleTest = pd.DataFrame(
    scaler.transform(sampleTest),
    columns=sampleTest.columns,
    index=sampleTest.index
)
print(sampleTest)
