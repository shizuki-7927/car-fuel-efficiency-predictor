import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
import os
import streamlit.components.v1 as components
from sklearn.metrics import mean_squared_error, r2_score
import japanize_matplotlib

st.title("🚗 車の燃費予測アプリ")

# ===== モデル =====
model = joblib.load("src/model.pkl")
scaler = joblib.load("src/scaler.pkl")

# ===== 入力 =====
cylinders = st.number_input("シリンダー数", 3, 12, 4)
displacement = st.number_input("排気量", 50, 500, 200)
horsepower = st.number_input("馬力", 40, 250, 100)
weight = st.number_input("重量", 1500, 5000, 2500)
acceleration = st.number_input("加速度", 5.0, 25.0, 15.0)
model_year = st.slider("モデル年式", 70, 82, 76)
origin = st.selectbox("製造国", [1, 2, 3], format_func=lambda x: {1:"USA",2:"Europe",3:"Japan"}[x])

if st.button("燃費を予測"):

    # ===== DataFrame =====
    X_new = pd.DataFrame([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]],
        columns=["cylinders","displacement","horsepower","weight","acceleration","model year","origin"]
    )

    # 特徴量追加
    X_new["power_to_weight"] = X_new["horsepower"] / X_new["weight"]
    X_new["disp_per_cyl"] = X_new["displacement"] / X_new["cylinders"]

    X_new = pd.get_dummies(X_new, columns=["origin"], drop_first=True)
    X_new.columns = X_new.columns.str.replace(".0", "", regex=False)

    X_new = X_new.reindex(
        columns=[
            "cylinders","displacement","weight","acceleration","model year",
            "power_to_weight","disp_per_cyl","origin_2","origin_3"
        ],
        fill_value=0
    )

    X_new_scaled = scaler.transform(X_new)
    mpg_pred = model.predict(X_new_scaled).item()

    st.success(f"予測燃費：{mpg_pred:.2f} MPG")

    # ===== 評価可視化 =====
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    y_pred = model.predict(X_test)

    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test.values.flatten(), y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_xlabel("実測値")
    ax.set_ylabel("予測値")
    ax.set_title("実測値 vs 予測値")
    st.pyplot(fig)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.write(f"RMSE: {rmse:.3f}")
    st.write(f"R²: {r2:.3f}")

    # ===== SHAP =====
    st.subheader("🧠 モデルの判断理由（SHAP Decision Plot）")
    if os.path.exists("outputs/decision_plot_example.png"):
        st.image("outputs/decision_plot_example.png", use_container_width=True)
    else:
        st.warning("SHAP画像がありません。先に shap_analysis.py を実行してください。")
