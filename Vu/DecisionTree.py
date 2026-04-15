from pandas import read_csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error, r2_score
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
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/3, random_state=42)


# Khởi tạo decision tree
model = DecisionTreeRegressor(
    # MSE và SDR có liên quan, thực chất tính MSE nhỏ nhất thì sẽ ra SDR lớn nhất
    criterion="squared_error",
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# Huấn luyện
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

plt.figure(figsize=(12, 8))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    rounded=True
)

plt.show()
# Tính các chỉ số:
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"MSE: {mse: .2f}")
print(f"R2: {r2: .2f}")
