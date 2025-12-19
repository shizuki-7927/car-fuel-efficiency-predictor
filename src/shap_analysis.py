import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# ===== 読み込み =====
model = joblib.load("src/model.pkl")
scaler = joblib.load("src/scaler.pkl")

X_train = pd.read_csv("data/processed/X_train.csv")
X_test  = pd.read_csv("data/processed/X_test.csv")

# ===== SHAP Explainer（Linear用）=====
explainer = shap.LinearExplainer(
    model,
    X_train,
    feature_perturbation="interventional"
)

shap_values = explainer(X_test)

# ===== 出力先 =====
os.makedirs("outputs", exist_ok=True)

# ===== summary plot =====
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=150)
plt.close()

print("✅ SHAP summary saved: outputs/shap_summary.png")
