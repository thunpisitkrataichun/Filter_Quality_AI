import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

import joblib
# Load the dataset

# Assuming the dataset is in CSV format and located at the specified path
data = pd.read_csv("dataset/StudentPerformanceFactors.csv")  # Replace with your actual file name

target = "Exam_Score" #Colume ที่ต้องการทำนาย

X = data.drop(columns=[target]) # ทุกเเทวยกเว้น target
Y = data[target] # แถวที่ต้องทำนาย


mapping = {
    "Low":0,
    "Medium":1,
    "High":2
}

X["Motivation_Level"] = X["Motivation_Level"].map(mapping)
#แปลงอักษรให้เป็นเลข 
le = LabelEncoder()

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

X_train,X_test, Y_train, Y_test= train_test_split(X , Y , test_size=0.2 , random_state=42)


# เลือกโมเดลที่ใช้

model = LinearRegression()

model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)



# สรุปผล
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)
print(data["Exam_Score"].max())

joblib.dump(model, "student_model.pkl")
