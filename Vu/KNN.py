from pandas import read_csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


energydata = read_csv("energydata_complete.csv", delimiter=",")
energydata['date'] = pd.to_datetime(energydata['date'])
energydata['Hour'] = energydata['date'].dt.hour
energydata['DayOfWeek'] = energydata['date'].dt.dayofweek

# Tách đặc trưng
X = energydata.drop(["Appliances", "date", "rv1",
                    "rv2", "Unnamed: 29"], axis=1)
# X = energydata.drop(["Appliances", "date", "Unnamed: 29"], axis=1)
Y = energydata["Appliances"]
# print(X)


# Phân chia tập dữ liệu theo nghi thức hold-out 2/3 1/3
# Vì muốn kiểm tra dữ liệu từ trên xuống dưới nên không dùng randomstate nữa
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/3, random_state=42, shuffle=True)

# # Khởi tạo scaler
scaler = MinMaxScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

# # Khởi tạo KNN hồi quy
k_values = range(1, 60, 2)
kBestForMAE = 1
kBestForMSE = 1
kBestForRMSE = 1

minMAE = 999999
minMSE = 999999
minRMSE = 999999

for k in k_values:
    modelKNN = KNeighborsRegressor(n_neighbors=k, p=2)
    modelKNN.fit(X_train_scaler, Y_train)
    Y_pred = modelKNN.predict(X_test_scaler)
    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    if (mae < minMAE):
        minMAE = mae
        kBestForMAE = k
    if (mse < minMSE):
        minMSE = mse
        kBestForMSE = k
    if (rmse < minRMSE):
        minRMSE = rmse
        kBestForRMSE = k

print(f"Min Mean Absolute Error: {minMAE}")
print(f"K best for MAE: {kBestForMAE}")

print(f"Min Mean Square Error: {minMSE}")
print(f"K best for MSE: {kBestForMSE}")

print(f"Min Root Mean Square Error: {minRMSE}")
print(f"K best for RMSE: {kBestForRMSE}")
