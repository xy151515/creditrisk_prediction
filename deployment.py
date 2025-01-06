import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import shap

# Load the trained model and scaler
model = joblib.load("stacked_model.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.title("Credit Risk Prediction Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose an option:", ["Home", "Upload New Data", "Monitor Performance", "SHAP Explainability"])

# Home Section
if app_mode == "Home":
    st.header("Welcome to the Credit Risk Prediction App")
    st.write("This app helps with credit risk prediction, performance monitoring, and explainability.")

# Upload New Data Section
elif app_mode == "Upload New Data":
    st.header("Upload New Data")
    uploaded_file = st.file_uploader("Upload a CSV file for predictions", type="csv")
    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        new_data_scaled = scaler.transform(new_data)
        
        # Predictions and probabilities
        predictions = model.predict(new_data_scaled)
        probabilities = model.predict_proba(new_data_scaled)[:, 1]

        # Show predictions
        st.subheader("Predictions")
        new_data["Prediction"] = predictions
        new_data["Probability"] = probabilities
        st.write(new_data)

        # Download predictions
        st.download_button("Download Predictions", new_data.to_csv(index=False), file_name="predictions.csv")

# Monitor Performance Section
elif app_mode == "Monitor Performance":
    st.header("Model Performance Monitoring")
    test_data = pd.read_csv("test_data.csv")
    actual_labels = test_data.pop("actual_labels")

    # Preprocess test data
    test_data_scaled = scaler.transform(test_data)
    predictions = model.predict(test_data_scaled)
    probabilities = model.predict_proba(test_data_scaled)[:, 1]

    # Metrics
    accuracy = accuracy_score(actual_labels, predictions)
    precision = precision_score(actual_labels, predictions)
    recall = recall_score(actual_labels, predictions)
    auc_roc = roc_auc_score(actual_labels, probabilities)

    # Display metrics
    st.subheader("Metrics")
    st.metric("Accuracy", f"{accuracy:.4f}")
    st.metric("Precision", f"{precision:.4f}")
    st.metric("Recall", f"{recall:.4f}")
    st.metric("AUC-ROC", f"{auc_roc:.4f}")

    # Alert if metrics drop below thresholds
    thresholds = {"accuracy": 0.85, "precision": 0.80, "recall": 0.75, "auc_roc": 0.90}
    if accuracy < thresholds["accuracy"] or auc_roc < thresholds["auc_roc"]:
        st.error("Performance drop detected! Retrain the model.")

# SHAP Explainability Section
elif app_mode == "SHAP Explainability":
    st.header("SHAP Explainability")
    test_data = pd.read_csv("test_data.csv").drop("actual_labels", axis=1)
    shap_sample = test_data.sample(100, random_state=42)
    shap_sample_scaled = scaler.transform(shap_sample)

    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_sample_scaled)

    # SHAP summary plot
    st.subheader("SHAP Summary Plot")
    shap.summary_plot(shap_values, shap_sample, plot_type="bar", feature_names=test_data.columns)
    st.pyplot()
