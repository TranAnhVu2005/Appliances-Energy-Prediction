import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(
    "D:/All/Information Technology  - CTU/CurrentSemester/CT294 - Applied machine learning/MayHocUngDung_TeamWork/Appliances-Energy-Prediction/appliances+energy+prediction/energydata_complete.csv", sep=",")


# Thiết lập kích thước khung hình
plt.figure(figsize=(10, 6))

# Vẽ biểu đồ Histogram kết hợp đường cong mật độ (KDE) cho nhãn Appliances
sns.histplot(df['Appliances'], bins=50, kde=True, color='blue')
# Thêm tiêu đề
plt.title('Phân phối lượng điện tiêu thụ của thiết bị (Appliances)')
plt.xlabel('Lượng điện tiêu thụ (Wh)')
plt.ylabel('Tần suất (Số lượng dòng)')

# Hiển thị biểu đồ
plt.show()
