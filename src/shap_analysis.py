# src/train_model.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from data_preprocessing import preprocess_features, BASE_DIR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

def train_and_save():
    df = pd.read_csv(BASE_DIR / "data/raw/auto-mpg.csv")

    df = remove_outliers(df, "mpg")
    y_train = df["mpg"]
    X_train = df.drop(columns=["mpg"])

    X_scaled, scaler, _, cols = preprocess_features(X_train, fit_scaler=True)

    model = XGBRegressor(random_state=42)

    param_grid = {
        "n_estimators": [300, 500],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    grid = GridSearchCV(model, param_grid, scoring="r2", cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_scaled, y_train)

    best_model = grid.best_estimator_
    joblib.dump(best_model, BASE_DIR / "src/model.pkl")

    print("✅ 学習 & 保存 完了！")

if __name__ == "__main__":
    train_and_save()
