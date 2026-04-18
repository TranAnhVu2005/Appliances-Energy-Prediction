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
             "rv2"], axis=1).values


y = df["Appliances"].values
# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Thử nhiều giá trị k
# k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 60]
k_values = range(1, 50, 2)
results = []

for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k)  # Mặc định là p=2
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    results.append((k, mae))

# In kết quả
for k, mae in results:
    print(f"k = {k}, MAE = {mae}")

# Vẽ biểu chọn k
k_list = []
mae_list = []
min_mae = 99999
kBestForMAE = 1

for k, mae in results:
    k_list.append(k)
    mae_list.append(mae)
    if (mae < min_mae):
        min_mae = mae
        kBestForMAE = k

plt.plot(k_list, mae_list, marker='o')
plt.xlabel("k")
plt.ylabel("MAE")
plt.title("KNN - chọn k tối ưu")
plt.show()


print(f"Min Mean Absolute Error: {min_mae}")
print(f"K best for MAE: {kBestForMAE}")
