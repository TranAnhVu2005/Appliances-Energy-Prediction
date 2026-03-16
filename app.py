import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv(
    "D:/MayHocUngDung_TeamWork/Appliances-Energy-Prediction/appliances+energy+prediction/energydata_complete.csv", sep=",")

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

# # df_number = df.select_dtypes(include=['number'])
# # print(df_number.T.head(10))
# # print(df_number.T.describe())

# # df_object = df.select_dtypes(include=['object'])
# # print(df_object.head(10))
# # print(df_object.describe())
# # print(f"Số hàng bị trùng lặp là: {df.duplicated().sum()}")
# # print(f"Số hàng bị null là:\n {df.isnull().sum()}")

# # Lấy các cột dạng số
# numeric_cols = df.select_dtypes(include=['number']).columns

# # Vẽ histogram cho từng feature
# for col in numeric_cols:
#     plt.figure(figsize=(6, 4))

#     sns.histplot(df[col], bins=30, kde=True)

#     plt.title(f"Histogram của {col}")
#     plt.xlabel(col)
#     plt.ylabel("Tần suất")

#     plt.show()
