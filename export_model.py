import numpy as np
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load dữ liệu
data_path = "D:/All/Information Technology  - CTU/CurrentSemester/CT294 - Applied machine learning/MayHocUngDung_TeamWork/Appliances-Energy-Prediction/appliances+energy+prediction/energydata_complete.csv"
print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path, sep=",")
df['date'] = pd.to_datetime(df['date'])
df['Hour'] = df['date'].dt.hour
df['DayOfWeek'] = df['date'].dt.dayofweek

# Tiền xử lý
X = df.drop(["Appliances", "date", "rv1", "rv2"], axis=1)
y = df["Appliances"].values

# Lấy ra tên các columns (feature names) để backend sử dụng
feature_names = list(X.columns)
print("Features used:", feature_names)

X = X.values

# Hold-out (Giữ nguyên như trong Combine.py)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# MinMaxScaler
print("Fitting MinMaxScaler...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Khai báo model
models = {
    "KNN": KNeighborsRegressor(n_neighbors=21),
    "Decision_Tree": DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=15,
        min_samples_split=40,
        min_samples_leaf=21,
        random_state=42
    ),
    "Linear_Regression": LinearRegression(),
    "Random_Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
}

# Tạo thư mục backend/models nếu chưa có
os.makedirs("backend/models", exist_ok=True)

# Lưu scaler
joblib.dump(scaler, "backend/models/scaler.joblib")
print("Saved scaler.joblib")

# Lưu feature names
joblib.dump(feature_names, "backend/models/feature_names.joblib")
print("Saved feature_names.joblib")

# Train & save models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f"backend/models/{name}.joblib")
    print(f"Saved {name}.joblib")

print("All models exported successfully!")
