import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ===== データ読み込み =====
df = pd.read_csv("data/raw/auto-mpg.csv")

# horsepower を数値に変換
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")

# 欠損除去
df = df.dropna()

# ===== 特徴量・目的変数 =====
X = df.drop("mpg", axis=1)
y = df["mpg"]

# ===== 特徴量工学 =====
X["power_to_weight"] = X["horsepower"] / X["weight"]
X["disp_per_cyl"] = X["displacement"] / X["cylinders"]

# ===== One-Hot Encoding =====
X = pd.get_dummies(X, columns=["origin"], drop_first=True)
X.columns = X.columns.str.replace(".0", "", regex=False)

# ===== 列順固定 =====
feature_cols = [
    "cylinders",
    "displacement",
    "weight",
    "acceleration",
    "model year",
    "power_to_weight",
    "disp_per_cyl",
    "origin_2",
    "origin_3"
]
X = X[feature_cols]

# ===== train / test split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== スケーリング =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== 保存 =====
os.makedirs("data/processed", exist_ok=True)

pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv("data/processed/X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=feature_cols).to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

joblib.dump(scaler, "src/scaler.pkl")

print("✅ data preprocessing complete")
