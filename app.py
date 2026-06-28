import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="LoanGuard AI", page_icon="🏦", layout="wide")

st.title("🏦 LoanGuard AI — Loan Default Risk Predictor")
st.markdown("*ML-powered tool to assess loan default risk based on borrower financial profile*")

@st.cache_data
def load_and_train():
    df = pd.read_csv("Default_Fin.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Defaulted?": "Defaulted"})

    X = df[["Employed", "Bank Balance", "Annual Salary"]]
    y = df["Defaulted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    return model, df, acc, auc, cm

model, df, acc, auc, cm = load_and_train()

# --- Metrics ---
st.markdown("### 📈 Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{acc*100:.1f}%")
col2.metric("ROC-AUC", f"{auc:.2f}")
col3.metric("Dataset Size", "10,000")
col4.metric("Default Rate", f"{df['Defaulted'].mean()*100:.1f}%")

st.markdown("---")

# --- Prediction ---
st.subheader("🔍 Predict Default Risk for a Borrower")

c1, c2, c3 = st.columns(3)
with c1:
    employed = st.selectbox("Employment Status", ["Employed", "Unemployed"])
with c2:
    bank_balance = st.number_input(
        "Bank Balance (₹)", min_value=0, max_value=100000,
        value=10000, step=500
    )
with c3:
    annual_salary = st.number_input(
        "Annual Salary (₹)", min_value=0, max_value=1000000,
        value=400000, step=10000
    )

if st.button("Predict Risk", type="primary"):
    emp_val = 1 if employed == "Employed" else 0
    input_df = pd.DataFrame(
        [[emp_val, bank_balance, annual_salary]],
        columns=["Employed", "Bank Balance", "Annual Salary"]
    )
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.markdown("#### Result")
    if pred == 1:
        st.error(f"⚠️ **High Default Risk** — Estimated Probability: {prob*100:.1f}%")
    else:
        st.success(f"✅ **Low Default Risk** — Estimated Probability: {prob*100:.1f}%")

    # Risk gauge
    fig_g, ax_g = plt.subplots(figsize=(6, 1))
    ax_g.barh(["Risk"], [prob], color="crimson" if pred == 1 else "seagreen", height=0.4)
    ax_g.barh(["Risk"], [1 - prob], left=[prob], color="#e0e0e0", height=0.4)
    ax_g.set_xlim(0, 1)
    ax_g.set_xlabel("Default Probability")
    ax_g.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax_g.set_title(f"Default Probability: {prob*100:.1f}%")
    st.pyplot(fig_g)

st.markdown("---")

# --- Feature Importance ---
st.subheader("📊 Feature Importance")
feat_imp = pd.Series(
    model.feature_importances_,
    index=["Employed", "Bank Balance", "Annual Salary"]
).sort_values(ascending=True)

fig1, ax1 = plt.subplots(figsize=(7, 2.5))
colors = ["#4C72B0", "#55A868", "#C44E52"]
ax1.barh(feat_imp.index, feat_imp.values, color=colors)
ax1.set_xlabel("Importance Score")
ax1.set_title("Which features drive default prediction?")
st.pyplot(fig1)

st.markdown("---")

# --- Data Distribution ---
st.subheader("📂 Dataset Overview")
tab1, tab2, tab3 = st.tabs(["Bank Balance", "Annual Salary", "Default by Employment"])

with tab1:
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    df[df["Defaulted"] == 0]["Bank Balance"].hist(bins=40, ax=ax2, alpha=0.6, label="No Default", color="seagreen")
    df[df["Defaulted"] == 1]["Bank Balance"].hist(bins=40, ax=ax2, alpha=0.6, label="Default", color="crimson")
    ax2.set_xlabel("Bank Balance (₹)")
    ax2.legend()
    st.pyplot(fig2)

with tab2:
    fig3, ax3 = plt.subplots(figsize=(7, 3))
    df[df["Defaulted"] == 0]["Annual Salary"].hist(bins=40, ax=ax3, alpha=0.6, label="No Default", color="seagreen")
    df[df["Defaulted"] == 1]["Annual Salary"].hist(bins=40, ax=ax3, alpha=0.6, label="Default", color="crimson")
    ax3.set_xlabel("Annual Salary (₹)")
    ax3.legend()
    st.pyplot(fig3)

with tab3:
    fig4, ax4 = plt.subplots(figsize=(5, 3))
    default_by_emp = df.groupby("Employed")["Defaulted"].mean() * 100
    default_by_emp.index = ["Unemployed", "Employed"]
    ax4.bar(default_by_emp.index, default_by_emp.values, color=["crimson", "seagreen"])
    ax4.set_ylabel("Default Rate (%)")
    ax4.set_title("Default Rate by Employment Status")
    st.pyplot(fig4)

st.markdown("---")
st.caption("LoanGuard AI | Built by Sagar Kaushik | Random Forest Classifier | Kaggle: Default_Fin Dataset | 10,000 records")
