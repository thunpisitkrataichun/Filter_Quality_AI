import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


model = joblib.load("models/random_forest/bmw_randomforest.pkl")
data = pd.read_csv("data/raw/bmw_global_sales_2018_2025.csv")

target = "Units_Sold"
x = data.drop(columns=target)
y = data[target]


for col in x.select_dtypes(include=object).columns :
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])
    
# เรื่มการ Train

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2 , random_state=42)

y_pred = model.predict(x_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

print(model.feature_importances_)