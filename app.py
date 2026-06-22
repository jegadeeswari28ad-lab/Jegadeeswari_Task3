import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("---")

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load("fraud_model.pkl")

# =====================================
# FILE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "Upload Credit Card Dataset",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # ==========================
    # PREPROCESS
    # ==========================

    scaler = StandardScaler()

    df["Amount"] = scaler.fit_transform(
        df[["Amount"]]
    )

    df["Time"] = scaler.fit_transform(
        df[["Time"]]
    )

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ==========================
    # KPI CARDS
    # ==========================

    total = len(df)
    fraud = len(df[df["Class"] == 1])
    normal = len(df[df["Class"] == 0])

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Transactions", f"{total:,}")
    c2.metric("Fraud Cases", fraud)
    c3.metric("Normal Cases", normal)
    c4.metric("Fraud Rate", f"{fraud/total*100:.4f}%")

    st.markdown("---")

    # ==========================
    # DATA PREVIEW
    # ==========================

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ==========================
    # PIE CHART
    # ==========================

    st.subheader("Fraud Distribution")

    fig = px.pie(
        values=[normal, fraud],
        names=["Normal", "Fraud"],
        hole=0.4
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # AMOUNT DISTRIBUTION
    # ==========================

    st.subheader("Transaction Amount Distribution")

    fig = px.histogram(
        df,
        x="Amount",
        nbins=50
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # CORRELATION HEATMAP
    # ==========================

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(14,8))

    sns.heatmap(
        df.corr(),
        cmap="coolwarm",
        center=0,
        ax=ax
    )

    st.pyplot(fig)

    # ==========================
    # FEATURE IMPORTANCE
    # ==========================

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    fig = px.bar(
        importance.head(15),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Features"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # ROC CURVE
    # ==========================

    st.subheader("ROC Curve")

    probs = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probs
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    roc_fig = go.Figure()

    roc_fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"AUC = {roc_auc:.4f}"
        )
    )

    roc_fig.add_trace(
        go.Scatter(
            x=[0,1],
            y=[0,1],
            mode="lines",
            name="Random"
        )
    )

    roc_fig.update_layout(
        title="ROC Curve - Random Forest",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    st.plotly_chart(
        roc_fig,
        use_container_width=True
    )

    # ==========================
    # TRANSACTION PREDICTION
    # ==========================

    st.subheader("Predict Existing Transaction")

    row_id = st.number_input(
        "Select Transaction Row",
        0,
        len(df)-1,
        0
    )

    if st.button("Predict Transaction"):

        sample = X.iloc[row_id:row_id+1]

        pred = model.predict(sample)[0]

        if pred == 1:
            st.error("⚠ FRAUD TRANSACTION DETECTED")
        else:
            st.success("✅ LEGITIMATE TRANSACTION")

    # ==========================
    # REAL-TIME MONITORING
    # ==========================

    st.subheader("Real-Time Monitoring")

    if st.button("Start Monitoring"):

        placeholder = st.empty()

        sample_data = X.sample(15)

        for i in range(len(sample_data)):

            row = sample_data.iloc[i:i+1]

            pred = model.predict(row)[0]

            if pred == 1:

                placeholder.error(
                    f"🚨 Transaction {i+1}: FRAUD DETECTED"
                )

            else:

                placeholder.success(
                    f"✅ Transaction {i+1}: SAFE"
                )

            time.sleep(1)

    st.markdown("---")
    st.success("Fraud Detection Dashboard Loaded Successfully")