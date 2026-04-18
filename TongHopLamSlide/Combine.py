import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data

# df = pd.read_csv(
#     "D:/All/Information Technology  - CTU/CurrentSemester/CT294 - Applied machine learning/MayHocUngDung_TeamWork/Appliances-Energy-Prediction/appliances+energy+prediction/energydata_complete.csv", sep=",")

# X = df.drop("Appliances", axis=1)
# y = df["Appliances"]


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

# Hold-out
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Khai báo model
models = {
    "KNN": KNeighborsRegressor(n_neighbors=21),
    "Decision Tree": DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=15,
        min_samples_split=40,
        min_samples_leaf=21,
        random_state=42
    ),
    "Linear Regression": LinearRegression()
}

# Train & evaluate
mae_list = []
mse_list = []
rmse_list = []
r2_list = []
model_names = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)
    model_names.append(name)

    print(f"{name}: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

# VẼ BIỂU ĐỒ
plt.figure(figsize=(14, 10))

# Định nghĩa 3 màu tương ứng cho 3 mô hình (KNN, Decision Tree, Linear Regression)
colors = ['#3498db', '#2ecc71', '#e74c3c'] 

# MAE
plt.subplot(2, 2, 1)
plt.bar(model_names, mae_list, color=colors)
plt.title("Mean Absolute Error (MAE)", fontweight='bold')
plt.ylabel("Giá trị sai số")

# MSE
plt.subplot(2, 2, 2)
plt.bar(model_names, mse_list, color=colors)
plt.title("Mean Squared Error (MSE)", fontweight='bold')

# RMSE
plt.subplot(2, 2, 3)
plt.bar(model_names, rmse_list, color=colors)
plt.title("Root Mean Squared Error (RMSE)", fontweight='bold')
plt.ylabel("Giá trị sai số")

# R2
plt.subplot(2, 2, 4)
plt.bar(model_names, r2_list, color=colors)
plt.title("R2 Score", fontweight='bold')

plt.tight_layout()
plt.show()
