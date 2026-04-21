import pandas as pd
import math
import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression



energydata = read_csv("D:/MHUD_TeamWork/appliances+energy+prediction/energydata_complete.csv", sep=",")
energydata['date'] = pd.to_datetime(energydata['date'])
energydata['Hour'] = energydata['date'].dt.hour
energydata['DayOfWeek'] = energydata['date'].dt.dayofweek

# Tách đặc trưng
X = energydata.drop(["Appliances", "date", "rv1",
                    "rv2"], axis=1)
Y = energydata["Appliances"]


# Phân chia tập dữ liệu theo nghi thức hold-out
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=False)

# Vẽ biểu đồ phân tán
plt.scatter(energydata.RH_1, energydata.Appliances)
plt.show()

# # Khởi tạo scaler
scaler = MinMaxScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

# Khởi tạo mô hình
model = LinearRegression()
model.fit(X_train_scaler, Y_train)
Y_pred = model.predict(X_test_scaler)

print("Cac thuoc tinh:", X_train.columns.values)
print("He so cua cac thuoc tinh:", model.coef_)
print("Bias (intercept):", model.intercept_)

# Tính MAE
mae = mean_absolute_error(Y_test, Y_pred)

# Tính MSE
mse = mean_squared_error(Y_test, Y_pred)

# Tính rmse
rmse = math.sqrt(mse)

# TÍnh R2
r2 = r2_score(Y_test, Y_pred)

print("="*100)
print("Do chinh xac theo cac phuong phap đanh gia hieu qua giai thuat hoc cua bai toan hoi quy:")
print("="*100)
print("MAE:", mae)
print("R2:", r2)
print("MSE:", mse)
print("RMSE:", rmse)

plt.figure(figsize=(8, 5))
plt.scatter(Y_test, Y_pred, alpha=0.3)
plt.plot([Y_test.min(), Y_test.max()],
         [Y_test.min(), Y_test.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
