import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
import os

# ===== モデル読み込み =====
model = joblib.load("src/model.pkl")

# ===== データ読み込み =====
X_test = pd.read_csv("data/processed/X_test.csv")

# ===== OneHot Encoding =====
X_test = pd.get_dummies(X_test, columns=["origin"], drop_first=True)
X_test.columns = X_test.columns.str.replace(".0", "", regex=False)

# ===== 列順を学習時と一致 =====
X_test = X_test.reindex(
    columns=["cylinders","displacement","weight","acceleration","model year","origin_2","origin_3"],
    fill_value=0
)

X_test = X_test.astype(float)

# ===== SHAP Explainer =====
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# ===== Decision Plot =====
idx = 10

plt.figure(figsize=(12, 4))

shap.decision_plot(
    explainer.expected_value,
    shap_values[idx],
    X_test.iloc[idx, :],
    show=False   # ← これが超重要
)

plt.tight_layout()

# ===== 保存 =====
os.makedirs("outputs", exist_ok=True)
plt.savefig(
    "outputs/decision_plot_example.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("🎉 decision plot saved → outputs/decision_plot_example.png")
