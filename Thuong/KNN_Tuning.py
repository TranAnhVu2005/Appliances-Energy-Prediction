import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load dữ liệu
df = pd.read_csv("D:/MHUD_TeamWork/appliances+energy+prediction/energydata_complete.csv", sep=",")

# Tiền xử lý
df = df.drop(columns=["date", "rv1", "rv2"])

# Chia X, y
X = df.drop("Appliances", axis=1)
y = df["Appliances"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Thử nhiều giá trị k
k_values = [3, 5, 7, 9, 11, 13, 15, 17, 60]
results = []

for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k) # Mặc định là p=2
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

for k, mae in results:
    k_list.append(k)
    mae_list.append(mae)

plt.plot(k_list, mae_list, marker='o')
plt.xlabel("k")
plt.ylabel("MAE")
plt.title("KNN - chọn k tối ưu")
plt.show() # Chọn k=3 là tối ưu nhất do, MAE nhỏ nhất

# Train KNN
# knn = KNeighborsRegressor(n_neighbors=5)
# knn.fit(X_train_scaled, y_train)

# Dự đoán
# y_pred = knn.predict(X_test_scaled)

# Đánh giá
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)


# print("KNN Results:")
# print("MAE:", mae)
# print("MSE:", mse)
# print("RMSE:", rmse)
