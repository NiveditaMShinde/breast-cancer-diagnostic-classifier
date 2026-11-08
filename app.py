import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Breast Cancer Diagnostic Classifier", layout="wide")

st.title("Breast Cancer Wisconsin (Diagnostic) — Model Benchmark")
st.markdown("Interactive demonstration of 5 trained classification models evaluated on held-out test data without retraining.")

# Model file mappings
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
}

@st.cache_resource
def load_model_and_scaler(model_name):
    model = joblib.load(MODEL_FILES[model_name])
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

# Sidebar controls
st.sidebar.header("User Controls")
selected_model_name = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))
uploaded_file = st.sidebar.file_uploader("Upload Test CSV (test_data.csv)", type=["csv"])

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded: {df_test.shape[0]} rows, {df_test.shape[1]} columns")

    # Check for target column
    if "target" not in df_test.columns:
        st.error("Uploaded CSV must contain the 'target' column.")
    else:
        X_test = df_test.drop(columns=["target"])
        y_test = df_test["target"]

        # Load saved artifacts
        model, scaler = load_model_and_scaler(selected_model_name)

        # Scale features only for distance/margin models
        if selected_model_name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test.values

        # Predictions (without retraining)
        y_pred = model.predict(X_eval)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_eval)[:, 1]
        else:
            y_prob = y_pred

        # Compute the 6 required evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Display metric cards
        st.subheader(f"Evaluation Metrics: {selected_model_name}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("AUC", f"{auc:.4f}")
        c3.metric("Precision", f"{prec:.4f}")
        c4.metric("Recall", f"{rec:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC", f"{mcc:.4f}")

        st.markdown("---")

        # Visualizations: Confusion Matrix & Classification Report
        col_cm, col_report = st.columns(2)

        with col_cm:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap="Blues", cbar=False,
                xticklabels=["Malignant (0)", "Benign (1)"],
                yticklabels=["Malignant (0)", "Benign (1)"]
            )
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

        with col_report:
            st.subheader("Classification Report")
            report_dict = classification_report(
                y_test, y_pred, 
                target_names=["Malignant (0)", "Benign (1)"], 
                output_dict=True
            )
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format("{:.4f}"))
else:
    st.info("Please upload `test_data.csv` using the sidebar file uploader to evaluate the models.")