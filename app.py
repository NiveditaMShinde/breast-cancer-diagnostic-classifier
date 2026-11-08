import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Diagnostic Classifier",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Modern CSS Styling for UI Polish
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Metric card container styling */
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    div[data-testid="stMetricLabel"] > div {
        color: #64748b;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] > div {
        color: #0f172a;
        font-size: 1.65rem;
        font-weight: 700;
    }
    /* Clean divider spacing */
    hr {
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Model File Registry
# -----------------------------------------------------------------------------
MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}

# -----------------------------------------------------------------------------
# Cached Artifact Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_scaler():
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model(model_filename):
    model_path = os.path.join(MODEL_DIR, model_filename)
    with open(model_path, "rb") as f:
        return pickle.load(f)

# -----------------------------------------------------------------------------
# Sidebar: Controls & Developer Info
# -----------------------------------------------------------------------------
st.sidebar.header("User Controls")

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_FILES.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (test_data.csv)",
    type=["csv"]
)

# -----------------------------------------------------------------------------
# Main Header Section
# -----------------------------------------------------------------------------
st.title("Breast Cancer Diagnostic Classifier")
st.caption("Developed by: Nivedita Shinde")
st.markdown("Interactive demonstration of 5 trained classification models evaluated on held-out test data without retraining.")

st.markdown("---")

# -----------------------------------------------------------------------------
# Data Processing & Evaluation
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Assume the last column or 'target' column is the ground truth
        if "target" in df.columns:
            y_test = df["target"]
            X_test = df.drop(columns=["target"])
        else:
            y_test = df.iloc[:, -1]
            X_test = df.iloc[:, :-1]

        # Standardize features using the pre-fit scaler artifact
        scaler = load_scaler()
        X_test_scaled = scaler.transform(X_test)

        # Load selected model and run inference
        model = load_model(MODEL_FILES[selected_model_name])
        y_pred = model.predict(X_test_scaled)

        # Compute probability scores for AUC calculation
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test_scaled)
            y_proba = 1 / (1 + np.exp(-scores))
        else:
            y_proba = y_pred

        # ---------------------------------------------------------------------
        # Evaluation Metrics Computation
        # ---------------------------------------------------------------------
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader(f"Evaluation Metrics: {selected_model_name}")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{acc:.4f}")
        m2.metric("AUC", f"{auc:.4f}")
        m3.metric("Precision", f"{prec:.4f}")
        m4.metric("Recall", f"{rec:.4f}")
        m5.metric("F1 Score", f"{f1:.4f}")
        m6.metric("MCC", f"{mcc:.4f}")

        st.markdown("---")

        # ---------------------------------------------------------------------
        # Confusion Matrix & Classification Report Visualizations
        # ---------------------------------------------------------------------
        col_plot, col_report = st.columns([1, 1.2])

        with col_plot:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4.5, 3.8), dpi=150)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                square=True,
                linewidths=1.5,
                linecolor="white",
                annot_kws={"size": 13, "weight": "bold"},
                ax=ax
            )
            ax.set_xlabel("Predicted Label", fontsize=9, fontweight="semibold")
            ax.set_ylabel("True Label", fontsize=9, fontweight="semibold")
            ax.set_xticklabels(["Malignant (0)", "Benign (1)"], fontsize=8)
            ax.set_yticklabels(["Malignant (0)", "Benign (1)"], fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)

        with col_report:
            st.subheader("Classification Report")
            report_dict = classification_report(
                y_test,
                y_pred,
                target_names=["Malignant (0)", "Benign (1)"],
                output_dict=True,
                zero_division=0
            )
            report_df = pd.DataFrame(report_dict).transpose()
            formatted_report = report_df.map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
            st.dataframe(formatted_report, use_container_width=True)

    except Exception as e:
        st.error(f"Error evaluating model on provided data: {e}")

else:
    st.info("👈 Please upload `test_data.csv` from the sidebar to inspect model performance.")

# -----------------------------------------------------------------------------
# Sidebar Footer
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Breast Cancer ML Benchmark System")