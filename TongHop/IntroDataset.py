import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# Load dữ liệu
df = pd.read_csv(
    "D:/All/Information Technology  - CTU/CurrentSemester/CT294 - Applied machine learning/MayHocUngDung_TeamWork/Appliances-Energy-Prediction/appliances+energy+prediction/energydata_complete.csv", sep=",")
df['date'] = pd.to_datetime(df['date'])
df['Hour'] = df['date'].dt.hour
df['DayOfWeek'] = df['date'].dt.dayofweek


# Tiền xử lý
X = df.drop(["Appliances", "date", "rv1",
             "rv2"], axis=1)


Y = df["Appliances"]
# Train/Test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=False
)

# Cho train
sample = X_train.iloc[:, :10].sample(10, random_state=42)
samplelabel = Y_train.loc[sample.index]

# scale hoặc không, comment hoặc gỡ comment dòng này
scaler = MinMaxScaler()
sample = pd.DataFrame(
    scaler.fit_transform(sample),
    columns=sample.columns,
    index=sample.index
)

# sample.insert(0, "Appliances", samplelabel)
# print(sample)

# Cho test
sampleTest = X_test.iloc[:, :10].sample(2, random_state=42)
# scale hoặc không, comment hoặc gỡ comment dòng này
sampleTest = pd.DataFrame(
    scaler.transform(sampleTest),
    columns=sampleTest.columns,
    index=sampleTest.index
)
# print(sampleTest)


# Tính toán check lại đáp án
array_train = sample.values
array_test = sampleTest.values
# 1. Lấy ra tọa độ (vector đặc trưng) của X1 và X2 từ tập Test
X1 = array_test[0]
X2 = array_test[1]

# 2. Tính khoảng cách Euclidean từ X1, X2 đến toàn bộ 10 điểm trong array_train
dist_X1 = np.sqrt(np.sum((array_train - X1)**2, axis=1))
dist_X2 = np.sqrt(np.sum((array_train - X2)**2, axis=1))

# 3. Tạo danh sách tên điểm (A -> K)
point_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K']

# 4. Gom kết quả vào DataFrame và làm tròn 6 chữ số thập phân
result_table = pd.DataFrame({
    'Điểm': point_names,
    'Khoảng cách đến X1': np.round(dist_X1, 6),  # Sửa số 3 thành số 6 ở đây
    'Khoảng cách đến X2': np.round(dist_X2, 6),  # Sửa số 3 thành số 6 ở đây
    'Nhãn y': samplelabel.values
})

print("--- BẢNG KHOẢNG CÁCH TỪ 2 ĐIỂM TEST ĐẾN 10 ĐIỂM TRAIN (6 SỐ THẬP PHÂN) ---")
print(result_table.to_string(index=False))

# Tự động tìm K=3 điểm gần X1 nhất
k_nearest_X1 = result_table.sort_values(by='Khoảng cách đến X1').head(3)

print("\n--- 3 ĐIỂM GẦN X1 NHẤT ---")
print(k_nearest_X1.to_string(index=False))

# Dự đoán cho X1
predict_X1 = k_nearest_X1['Nhãn y'].mean()
print(f"\n=> Lượng điện dự đoán cho X1 là: {predict_X1} Wh")
