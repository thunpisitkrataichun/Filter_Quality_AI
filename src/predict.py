import pandas as pd
import joblib

# ======================
# LOAD MODEL
# ======================
model = joblib.load("models/random_forest/bmw_randomforest.pkl")

# ======================
# INPUT DATA (แก้ค่าตรงนี้)
# ======================
new_data = pd.DataFrame([{
    "Year": 2023,
    "Month": 6,
    "Region": "Asia",
    "Model": "X5",
    "Avg_Price_EUR": 65000,
    "Revenue_EUR": 150000000,
    "BEV_Share": 0.25,
    "Premium_Share": 0.60,
    "GDP_Growth": 3.2,
    "Fuel_Price_Index": 1.1
}])
# ======================
# PREDICT
# ======================
result = model.predict(new_data)

print("Prediction:", result)

