import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

data = pd.read_csv(r"C:/Users/trant/OneDrive/Máy tính/MHUD/Pj/Data/energydata_complete.csv")

data["date"] = pd.to_datetime(data["date"])
data["hour"] = data["date"].dt.hour
data = data.drop("date", axis=1)

X = data.drop(["Appliances","rv1","rv2"], axis=1)
y = data["Appliances"]

kf = KFold(n_splits=9, shuffle= True, random_state=42)

mse_scores = []
rmse_scores = []
r2_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = DecisionTreeRegressor()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test, y_pred)

    mse_scores.append(mse)
    rmse_scores.append(rmse)
    r2_scores.append(r2)

    print(f"MSE: {mse}")







