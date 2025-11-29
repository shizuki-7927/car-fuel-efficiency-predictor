# src/train_model.py
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from xgboost import XGBRegressor


def remove_outliers(df, col):
    """IQR法で外れ値除去"""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]


def train_and_evaluate():

    # ----------------------
    # ① データ読み込み
    # ----------------------
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")["mpg"]
    y_test = pd.read_csv("data/processed/y_test.csv")["mpg"]

    # ----------------------
    # ② 外れ値除去（MPGベース）
    # ----------------------
    df_train = X_train.copy()
    df_train["mpg"] = y_train
    df_train = remove_outliers(df_train, "mpg")

    y_train = df_train["mpg"]
    X_train = df_train.drop("mpg", axis=1)

    # ----------------------
    # ③ OneHot Encoding（origin）
    # ----------------------
    X_train["origin"] = X_train["origin"].astype(int)
    X_train = pd.get_dummies(X_train, columns=["origin"], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=["origin"], drop_first=True)

    # 列ズレ対策（train基準）
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # ----------------------
    # ④ 特徴量選択（horsepower削除）
    # ----------------------
    if "horsepower" in X_train.columns:
        X_train = X_train.drop(columns=["horsepower"])
        X_test = X_test.drop(columns=["horsepower"])

    # ----------------------
    # ⑤ RobustScaler でスケーリング（目的変数はスケールしない）
    # ----------------------
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, "src/scaler.pkl")

    # ----------------------
    # ⑥ XGBoost モデル + GridSearchCV（ハイパーパラメータ最適化）
    # ----------------------
    model = XGBRegressor(random_state=42)

    param_grid = {
        "n_estimators": [300, 500],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    grid = GridSearchCV(
        model, param_grid,
        scoring="r2",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    print("\n🔥 最適なモデルパラメータ:")
    print(grid.best_params_)

    # ----------------------
    # ⑦ モデル評価
    # ----------------------
    y_pred = best_model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📈 モデル評価結果 (XGBoost + 最適化)")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"R²   : {r2:.3f}")

    # ----------------------
    # ⑧ モデル保存
    # ----------------------
    joblib.dump(best_model, "src/model.pkl")
    print("\n💾 モデルを src/model.pkl に保存しました")


if __name__ == "__main__":
    train_and_evaluate()
