import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ======================
# LOAD DATA
# ======================
data = pd.read_csv("data/raw/bmw_global_sales_2018_2025.csv")

target = "Units_Sold"
X = data.drop(columns=[target])
y = data[target]

# ======================
# SPLIT COLUMN TYPES
# ======================
categorical_cols = X.select_dtypes(include="object").columns
numeric_cols = X.select_dtypes(exclude="object").columns

# ======================
# PREPROCESSOR
# ======================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# ======================
# MODEL PIPELINE
# ======================
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# ======================
# TRAIN / TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# TRAIN
# ======================
model.fit(X_train, y_train)

# ======================
# PREDICT
# ======================
y_pred = model.predict(X_test)

# ======================
# EVALUATE
# ======================
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("## Model train Success ##")
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)


# ดึง feature importance
ohe = model.named_steps["preprocessor"].named_transformers_["cat"]
cat_features = ohe.get_feature_names_out(categorical_cols)

all_features = list(cat_features) + list(numeric_cols)
importances = model.named_steps["regressor"].feature_importances_

feature_df = pd.DataFrame({
    "feature": all_features,
    "importance": importances
}).sort_values(by="importance", ascending=True)

# plot
plt.figure()
plt.barh(feature_df["feature"], feature_df["importance"])
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
# ======================
# FEATURE IMPORTANCE
# ======================
# ดึง feature name หลัง OneHot
ohe = model.named_steps["preprocessor"].named_transformers_["cat"]
cat_features = ohe.get_feature_names_out(categorical_cols)

all_features = list(cat_features) + list(numeric_cols)

importances = model.named_steps["regressor"].feature_importances_

feature_df = pd.DataFrame({
    "feature": all_features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop Features:")
print(feature_df.head(10))

# ======================
# SAVE MODEL
# ======================
os.makedirs("models/random_forest", exist_ok=True)
joblib.dump(model, "models/random_forest/bmw_randomforest.pkl")

print("Model saved!")