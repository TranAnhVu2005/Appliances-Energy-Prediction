import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


# Load dữ liệu
df = pd.read_csv(
    "D:/All/Information Technology  - CTU/CurrentSemester/CT294 - Applied machine learning/MayHocUngDung_TeamWork/Appliances-Energy-Prediction/appliances+energy+prediction/energydata_complete.csv", sep=",")
df['date'] = pd.to_datetime(df['date'])
df['Hour'] = df['date'].dt.hour
df['DayOfWeek'] = df['date'].dt.dayofweek


# Tiền xử lý
X = df.drop(["date", "rv1",
             "rv2"], axis=1)

# sample = df[['date', 'Hour', 'DayOfWeek']].sample(n=10, random_state=42)
# print(sample.to_string(index=False))

# Lấy sample
X_sample = X.sample(n=12, random_state=42)

# Lấy 10 cột đầu (bao gồm Appliances nếu nó nằm đầu)
X_sample = X_sample.iloc[:, :11]

# Tách Appliances ra
y_sample = X_sample['Appliances']
X_features = X_sample.drop(columns=['Appliances'])

# Scale chỉ features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_features)

# Convert lại DataFrame
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=X_features.columns,
    index=X_sample.index
)

# Ghép lại: Appliances ở đầu
final_df = pd.concat([X_scaled_df], axis=1)

print(final_df[10:12])


# y = df["Appliances"].values
# # Train/Test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, shuffle=False
# )

# Chuẩn hóa dữ liệu
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Thử nhiều giá trị k
# # k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 60]
# k_values = range(1, 100, 2)
# results = []

# for k in k_values:
#     model = KNeighborsRegressor(n_neighbors=k)  # Mặc định là p=2
#     model.fit(X_train_scaled, y_train)
#     y_pred = model.predict(X_test_scaled)

#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     results.append((k, mae, mse, rmse))

# # In kết quả
# for k, mae, mse, rmse in results:
#     print(f"k = {k}, MAE = {mae}, MSE = {mse}, RMSE = {rmse}")

# # Vẽ biểu chọn k
# k_list = []
# mae_list = []
# mse_list = []
# rmse_list = []
# min_mae = 99999
# min_mse = 99999
# min_rmse = 99999
# kBestForMAE = 1
# kBestForMSE = 1
# KBestFOrRMSE = 1

# for k, mae, mse, rmse in results:
#     k_list.append(k)
#     mae_list.append(mae)
#     if (mae < min_mae):
#         min_mae = mae
#         kBestForMAE = k

# plt.plot(k_list, mae_list, marker='o')
# plt.xlabel("k")
# plt.ylabel("MAE")
# plt.title("KNN - chọn k tối ưu")
# plt.show()

# k_list = []

# for k, mae, mse, rmse in results:
#     k_list.append(k)
#     mse_list.append(mse)
#     if (mse < min_mse):
#         min_mse = mse
#         kBestForMSE = k

# plt.plot(k_list, mse_list, marker='o')
# plt.xlabel("k")
# plt.ylabel("MSE")
# plt.title("KNN - chọn k tối ưu")
# plt.show()

# k_list = []

# for k, mae, mse, rmse in results:
#     k_list.append(k)
#     rmse_list.append(rmse)
#     if (rmse < min_rmse):
#         min_rmse = rmse
#         KBestFOrRMSE = k

# plt.plot(k_list, rmse_list, marker='o')
# plt.xlabel("k")
# plt.ylabel("RMSE")
# plt.title("KNN - chọn k tối ưu")
# plt.show()


# print(f"Min Mean Absolute Error: {min_mae}")
# print(f"K best for MAE: {kBestForMAE}")

# print(f"Min Mean Square Error: {min_mse}")
# print(f"K best for MSE: {kBestForMSE}")

# print(f"Min Root Mean Square Error: {min_rmse}")
# print(f"K best for RMSE: {KBestFOrRMSE}")
