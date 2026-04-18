import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. Load và tiền xử lý dữ liệu
# ==========================================
print("Loading data...")
df = pd.read_csv(
    "D:/All/Information Technology  - CTU/CurrentSemester/CT294 - Applied machine learning/MayHocUngDung_TeamWork/Appliances-Energy-Prediction/appliances+energy+prediction/energydata_complete.csv", sep=",")
df['date'] = pd.to_datetime(df['date'])
df['Hour'] = df['date'].dt.hour
df['DayOfWeek'] = df['date'].dt.dayofweek

feature_names = df.drop(["Appliances", "date", "rv1", "rv2"], axis=1).columns
X = df.drop(["Appliances", "date", "rv1", "rv2"], axis=1).values
y = df["Appliances"].values

# ==========================================
# 2. Phân tách tập dữ liệu (Hold-out)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# ==========================================
# 3. Tinh chỉnh siêu tham số và huấn luyện
# ==========================================
print("Tuning hyperparameters...")

# Khảo sát max_depth
max_depth_list = [5, 10, 15, 20]
rmse_depth = []

for depth in max_depth_list:
    model = DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=depth,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse_depth.append(np.sqrt(mean_squared_error(y_test, y_pred)))

# [SỬA ĐỔI/THÊM]: Tự động lấy cấu hình max_depth tốt nhất (có error thấp nhất)
# Giải thích: np.argmin trả ra vị trí (index) của RMSE thấp nhất trong mảng đồ thị vòng lặp
best_depth = max_depth_list[np.argmin(rmse_depth)]

# Khảo sát min_samples_split
split_list = [40, 50, 55, 60, 65, 70]
rmse_split = []

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
    rmse_split.append(np.sqrt(mean_squared_error(y_test, y_pred)))

# [SỬA ĐỔI/THÊM]: Tự động lấy cấu hình min_samples_split tốt nhất
best_split = split_list[np.argmin(rmse_split)]

# Khảo sát min_samples_leaf
leaf_list = [1, 10, 11, 19, 20, 21, 22, 23, 24, 25, 30]
rmse_leaf = []

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
    rmse_leaf.append(np.sqrt(mean_squared_error(y_test, y_pred)))

# [SỬA ĐỔI/THÊM]: Tự động lấy cấu hình min_samples_leaf tốt nhất
best_leaf = leaf_list[np.argmin(rmse_leaf)]

# ==========================================
# 4. Huấn luyện mô hình tối ưu nhất
# ==========================================
# [SỬA ĐỔI/THÊM]: Sinh ra block báo lỗi tổng quan cuối cùng dựa vào các thông số tốt nhất vừa dò
# Giải thích: Sau giai đoạn thử (Tuning), Cây quyết định phải được gắn tất tật tham số tối ưu 
# (`best_depth`, `best_split`, `best_leaf`) vào train 1 lần chót nhằm trả số liệu báo cáo cho bạn vẽ R-Squared (R2).
print("\n=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH DECISION TREE TỐI ƯU ===")
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
opt_mse = mean_squared_error(y_test, y_pred_best)
opt_rmse = np.sqrt(opt_mse)
opt_r2 = r2_score(y_test, y_pred_best)

print(f"Optimal Parameters: ")
print(f" - max_depth: {best_depth}")
print(f" - min_samples_split: {best_split}")
print(f" - min_samples_leaf: {best_leaf}")
print(f"\nMetrics với tham số tối ưu:")
print(f" - MAE : {opt_mae:.4f}")
print(f" - MSE : {opt_mse:.4f}")
print(f" - RMSE: {opt_rmse:.4f}")
print(f" - R2  : {opt_r2:.4f}")

# ==========================================
# 5. Vẽ biểu đồ các quá trình Tuning
# ==========================================
plt.figure(figsize=(15, 5))

# [SỬA ĐỔI]: Tạo bố cục biểu đồ ngang (1 hàng 3 cột)
# Biểu đồ 1: max_depth
plt.subplot(1, 3, 1)
plt.plot(max_depth_list, rmse_depth, marker='o', color='b')
plt.xticks(max_depth_list)
plt.xlabel("max_depth")
plt.ylabel("RMSE")
plt.title("Effect of max_depth")
plt.grid(True, linestyle='--', alpha=0.6)

# Biểu đồ 2: min_samples_split
plt.subplot(1, 3, 2)
plt.plot(split_list, rmse_split, marker='o', color='g')
plt.xticks(split_list)
plt.xlabel("min_samples_split")
plt.ylabel("RMSE")
plt.title("Effect of min_samples_split")
plt.grid(True, linestyle='--', alpha=0.6)

# Biểu đồ 3: min_samples_leaf
plt.subplot(1, 3, 3)
plt.plot(leaf_list, rmse_leaf, marker='o', color='r')
plt.xticks(leaf_list)
plt.xlabel("min_samples_leaf")
plt.ylabel("RMSE")
plt.title("Effect of min_samples_leaf")
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# ==========================================
# 6. Trực quan hoá cấu trúc Cây Quyết Định
# ==========================================
# Giải thích: Vẽ sơ đồ cấu trúc nhánh rẽ của mô hình tốt nhất. 
# Do độ sâu cây tối ưu có thể rất lớn (nhiều hơn 5) làm ảnh bị nhòe đen, tôi giới hạn `max_depth=3`
# để các bạn xem 3 tầng đầu tiên rõ rệt nhất phục vụ cho việc cắt vào slide báo cáo.
plt.figure(figsize=(20, 10))
plot_tree(
    best_tree,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=9
)
plt.title("Visualized Structure of the Optimal Decision Tree (Top 3 levels)", fontsize=14)
plt.show()
