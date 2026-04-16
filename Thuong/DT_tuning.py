import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("D:/MHUD_TeamWork/appliances+energy+prediction/energydata_complete.csv", sep=",")

# Tiền xử lý
df = df.drop(columns=["date", "rv1", "rv2"])

X = df.drop("Appliances", axis=1)
y = df["Appliances"]

# Hold-out
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# max_depth
max_depth_list = [5, 10, 15, 20, None]
rmse_depth = []

for depth in max_depth_list:
    model = DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=depth,
        min_samples_split=100,
        min_samples_leaf=30,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse_depth.append(np.sqrt(mean_squared_error(y_test, y_pred)))

# min_samples_split
split_list = [2, 50, 55, 60, 65, 70, 100]
rmse_split = []

for split in split_list:
    model = DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=15,
        min_samples_split=split,
        min_samples_leaf=30,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse_split.append(np.sqrt(mean_squared_error(y_test, y_pred)))

# min_samples_leaf
leaf_list = [1, 10, 11, 19, 20, 21, 22, 23, 24, 25, 30]
rmse_leaf = []

for leaf in leaf_list:
    model = DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=15,
        min_samples_split=100,
        min_samples_leaf=leaf,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse_leaf.append(np.sqrt(mean_squared_error(y_test, y_pred)))

# Vẽ biểu đồ
plt.figure(figsize=(15, 5))

# Biểu đồ 1: max_depth
plt.subplot(1, 3, 1)
plt.plot(range(len(max_depth_list)), rmse_depth, marker='o')
plt.xticks(range(len(max_depth_list)), max_depth_list)
plt.xlabel("max_depth")
plt.ylabel("RMSE")
plt.title("max_depth")

# Biểu đồ 2: min_samples_split
plt.subplot(1, 3, 2)
plt.plot(range(len(split_list)), rmse_split, marker='o')
plt.xticks(range(len(split_list)), split_list)
plt.xlabel("min_samples_split")
plt.title("min_samples_split")


# Biểu đồ 3: min_samples_leaf
plt.subplot(1, 3, 3)
plt.plot(range(len(leaf_list)), rmse_leaf, marker='o')
plt.xticks(range(len(leaf_list)), leaf_list)
plt.xlabel("min_samples_leaf")
plt.title("min_samples_leaf")

plt.tight_layout()
plt.show()