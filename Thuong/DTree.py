import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
df = pd.read_csv("D:/MHUD_TeamWork/appliances+energy+prediction/energydata_complete.csv", sep=",")

# Tien xu ly
df = df.drop(columns=["date", "rv1", "rv2"])

X = df.drop("Appliances", axis=1).values
y = df["Appliances"].values

# KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_list = []
mae_list = []
r2_list = []

# Model (them tham so)
model = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth=15,
    min_samples_split=60,
    min_samples_leaf=23,
    random_state=42
)

# Loop tung fold
for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Chuan hoa
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Danh gia
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    rmse_list.append(rmse)
    mae_list.append(mae)
    r2_list.append(r2)

    print(f"Fold {fold}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, R2 = {r2:.4f}")

# Trung binh
print("\nTrung binh:")
print("RMSE:", np.mean(rmse_list))
print("MAE:", np.mean(mae_list))
print("R2:", np.mean(r2_list))