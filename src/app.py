# src/app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


st.title("🚗 車の燃費予測アプリ（改善版）")

# ===== モデル・スケーラー読み込み =====
model = joblib.load("src/model.pkl")
scaler = joblib.load("src/scaler.pkl")

# ===== 入力フォーム =====
cylinders = st.number_input("シリンダー数", min_value=3, max_value=12, value=4)
displacement = st.number_input("排気量 (cu inches)", min_value=50, max_value=500, value=200)
weight = st.number_input("車重 (lbs)", min_value=1500, max_value=5000, value=2500)
acceleration = st.number_input("加速度 (0-60mph)", min_value=5.0, max_value=25.0, value=15.0)
model_year = st.slider("モデル年式", 70, 82, 76)
origin = st.selectbox("製造国", ["USA", "Europe", "Japan"], index=0)


# ===== 予測処理 =====
if st.button("燃費を予測"):
    
    # OneHot Encoding
    origin_2 = 1 if origin == "Europe" else 0
    origin_3 = 1 if origin == "Japan" else 0

    X_new = pd.DataFrame([[
        cylinders, displacement, weight, acceleration, model_year, origin_2, origin_3
    ]], columns=["cylinders","displacement","weight","acceleration","model year","origin_2.0","origin_3.0"])

    # スケーリング
    X_new_scaled = scaler.transform(X_new)

    # 予測
    mpg_pred = model.predict(X_new_scaled).item()
    st.success(f"✨ 予測燃費: {mpg_pred:.2f} MPG")

    # ===== モデル性能可視化 =====

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")["mpg"]

    # OneHot化処理（trainと同じ処理）
    X_test = pd.get_dummies(X_test, columns=["origin"], drop_first=True)
    X_test = X_test.reindex(columns=X_new.columns, fill_value=0)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    # 散布図
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    sns.scatterplot(x=y_test.values.flatten(), y=y_pred.flatten(), ax=ax1)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax1.set_xlabel("実測値 MPG")
    ax1.set_ylabel("予測値 MPG")
    ax1.set_title("実測値 vs 予測値")
    st.pyplot(fig1)

    # 誤差ヒストグラム
    errors = y_test.values.flatten() - y_pred.flatten()
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.histplot(errors, bins=20, kde=True, ax=ax2)
    ax2.set_xlabel("誤差 (実測 - 予測)")
    ax2.set_title("予測誤差の分布")
    st.pyplot(fig2)

    # 数値評価
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.write(f"📊 RMSE: {rmse:.3f}")
    st.write(f"📈 R²スコア: {r2:.3f}")

import streamlit.components.v1 as components

force_html_path = "outputs/force_plot_example.html"
with open(force_html_path, "r", encoding="utf-8") as f:
    html_data = f.read()

components.html(html_data, height=300)