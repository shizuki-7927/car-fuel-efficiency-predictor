# src/train_model.py
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ===== データ読み込み =====
X_train = pd.read_csv("data/processed/X_train.csv")
X_test  = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv")["mpg"].values
y_test  = pd.read_csv("data/processed/y_test.csv")["mpg"].values

# ===== XGBoost モデル =====
model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# ===== 学習 =====
model.fit(X_train, y_train)

# ===== 評価 =====
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("✅ Model Evaluation (XGBoost)")
print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")
print(f"R²  : {r2:.3f}")

# ===== 保存 =====
joblib.dump(model, "src/model.pkl")
print("💾 XGBoost model saved")
