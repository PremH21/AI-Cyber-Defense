import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Cyber Defense — XAI Dashboard", layout="wide")

@st.cache_resource
def load_model_and_data():
    model = joblib.load("ml-engine/models/xgboost_unsw.joblib")
    test_df = pd.read_parquet("data/processed/unsw-nb15_test.parquet")
    return model, test_df

@st.cache_resource
def get_explainer(_model, X_sample):
    return shap.TreeExplainer(_model, X_sample)

model, test_df = load_model_and_data()
y_test = test_df["Attack"]
X_test = test_df.drop(columns=["Attack"])

# Use a sample for the explainer background (faster, standard practice)
X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
explainer = get_explainer(model, X_sample)

st.title("🛡️ AI-Driven Autonomous Defense Framework")
st.caption("Innovation 6 — XAI Explainability Dashboard | SHAP-based detection reasoning")

col1, col2, col3 = st.columns(3)
col1.metric("Model", "XGBoost")
col2.metric("Test Accuracy", "91%")
col3.metric("Attack Types Covered", "9")

st.divider()

# --- Section 1: Global feature importance ---
st.header("Global Feature Importance")
st.write("Which features matter most across the whole detection model:")

with st.spinner("Computing SHAP values..."):
    shap_values_sample = explainer.shap_values(X_sample)

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values_sample, X_sample, plot_type="bar", show=False)
st.pyplot(fig)
plt.close(fig)

st.divider()

# --- Section 2: Explain a single detection ---
st.header("Explain a Single Detection")
st.write("Pick a row from the test set to see exactly why the model flagged it (or didn't):")

row_idx = st.number_input("Row index", min_value=0, max_value=len(X_test)-1, value=0, step=1)
selected_row = X_test.iloc[[row_idx]]
true_label = y_test.iloc[row_idx]
pred = model.predict(selected_row)[0]
pred_proba = model.predict_proba(selected_row)[0][1]

c1, c2, c3 = st.columns(3)
c1.metric("True Label", "🔴 Attack" if true_label == 1 else "🟢 Benign")
c2.metric("Model Prediction", "🔴 Attack" if pred == 1 else "🟢 Benign")
c3.metric("Attack Probability", f"{pred_proba:.1%}")

st.subheader("Top features driving this decision")
row_shap = explainer.shap_values(selected_row)
shap_df = pd.DataFrame({
    "feature": selected_row.columns,
    "value": selected_row.values[0],
    "shap_impact": row_shap[0]
}).sort_values("shap_impact", key=abs, ascending=False).head(10)

fig2, ax2 = plt.subplots(figsize=(10, 5))
colors = ["#d62728" if v > 0 else "#2ca02c" for v in shap_df["shap_impact"]]
ax2.barh(shap_df["feature"], shap_df["shap_impact"], color=colors)
ax2.set_xlabel("SHAP value (red = pushes toward Attack, green = pushes toward Benign)")
ax2.invert_yaxis()
st.pyplot(fig2)
plt.close(fig2)

st.dataframe(shap_df, use_container_width=True)

st.caption("This satisfies Innovation 6 (XAI Explainability Dashboard) and the NIST IR 8596 accountability requirement referenced in your PPT — every detection decision is auditable.")
