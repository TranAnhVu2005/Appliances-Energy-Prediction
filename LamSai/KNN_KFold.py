import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
df = pd.read_csv(
    "D:/All/Information Technology  - CTU/CurrentSemester/CT294 - Applied machine learning/MayHocUngDung_TeamWork/Appliances-Energy-Prediction/appliances+energy+prediction/energydata_complete.csv", sep=",")
df['date'] = pd.to_datetime(df['date'])
df['Hour'] = df['date'].dt.hour
df['DayOfWeek'] = df['date'].dt.dayofweek


# Tiền xử lý
X = df.drop(["Appliances", "date", "rv1",
             "rv2"], axis=1).values

y = df["Appliances"].values

# KFold
kf = KFold(n_splits=10, shuffle=False)

mae_list = []
mse_list = []
rmse_list = []
r2_list = []

# Model
model = KNeighborsRegressor(n_neighbors=17)

# K - fold
# Loop từng fold
for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Chuẩn hóa
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Đánh giá
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)

    print(
        f"Fold {fold}: MAE = {mae:.4f}, MSE = {mse:.4f}, RMSE = {rmse:.4f}, R2 = {r2:.4f}")

# Trung bình
print("/nTrung binh:")
print("MAE:", np.mean(mae_list))
print("MSE:", np.mean(mse_list))
print("RMSE:", np.mean(rmse_list))
print("R2:", np.mean(r2_list))
