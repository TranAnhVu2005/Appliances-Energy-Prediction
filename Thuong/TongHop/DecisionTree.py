import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# 1. Load và tiền xử lý dữ liệu
df = pd.read_csv("D:/MHUD_TeamWork/appliances+energy+prediction/energydata_complete.csv", sep=",")
df['date'] = pd.to_datetime(df['date'])
df['Hour'] = df['date'].dt.hour
df['DayOfWeek'] = df['date'].dt.dayofweek

X = df.drop(["Appliances", "date", "rv1", "rv2"], axis=1).values
y = df["Appliances"].values

# 2. Phân tách tập dữ liệu (Hold-out)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 3. Tinh chỉnh siêu tham số và huấn luyện

# Khảo sát max_depth
max_depth_list = [5, 10, 15, 20]
mae_depth = []

for depth in max_depth_list:
    model = DecisionTreeRegressor(
        # Cắt sao tổng độ lệch chuẩn là nhỏ nhất so với độ lệch chuẩn ban đầu
        criterion='squared_error',
        max_depth=depth,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae_depth.append((mean_absolute_error(y_test, y_pred)))

# Giải thích: np.argmin trả ra vị trí (index) của mae thấp nhất trong mảng
best_depth = max_depth_list[np.argmin(mae_depth)]

# Khảo sát min_samples_split
split_list = [40, 50, 55, 60, 65, 70]
mae_split = []

for split in split_list:
    model = DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=None,
        min_samples_split=split,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae_split.append((mean_absolute_error(y_test, y_pred)))

best_split = split_list[np.argmin(mae_split)]

# Khảo sát min_samples_leaf
leaf_list = [13, 15, 19, 20, 21, 22, 23, 24, 25]
mae_leaf = []

for leaf in leaf_list:
    model = DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=leaf,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae_leaf.append((mean_absolute_error(y_test, y_pred)))

best_leaf = leaf_list[np.argmin(mae_leaf)]

# 4. Huấn luyện mô hình tối ưu nhất
print("\n=== KET QUA DANH GIA MO HINH DECISION TREE TOI UU ===")
best_tree = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth=best_depth,
    min_samples_split=best_split,
    min_samples_leaf=best_leaf,
    random_state=42
)
best_tree.fit(X_train, y_train)
y_pred_best = best_tree.predict(X_test)

opt_mae = mean_absolute_error(y_test, y_pred_best)


print(f"Optimal Parameters: ")
print(f" - max_depth: {best_depth}")
print(f" - min_samples_split: {best_split}")
print(f" - min_samples_leaf: {best_leaf}")
print(f"\nMetrics voi tham so toi uu:")
print(f" - MAE : {opt_mae:.4f}")


# 5. Vẽ biểu đồ các quá trình Tuning
plt.figure(figsize=(15, 5))  # Chiều rộng, chiều cao

# Tạo bố cục biểu đồ ngang (1 hàng 3 cột)
# Biểu đồ 1: max_depth
plt.subplot(1, 3, 1)
plt.plot(max_depth_list, mae_depth, marker='o', color='b')
plt.xticks(max_depth_list)
plt.xlabel("max_depth")
plt.ylabel("MAE")
plt.title("Effect of max_depth")
plt.grid(True, linestyle='--', alpha=0.6)

# Biểu đồ 2: min_samples_split
plt.subplot(1, 3, 2)
plt.plot(split_list, mae_split, marker='o', color='g')
plt.xticks(split_list)
plt.xlabel("min_samples_split")
plt.ylabel("MAE")
plt.title("Effect of min_samples_split")
plt.grid(True, linestyle='--', alpha=0.6)

# Biểu đồ 3: min_samples_leaf
plt.subplot(1, 3, 3)
plt.plot(leaf_list, mae_leaf, marker='o', color='r')
plt.xticks(leaf_list)
plt.xlabel("min_samples_leaf")
plt.ylabel("MAE")
plt.title("Effect of min_samples_leaf")
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
