import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
import shap
import numpy as np

# Load the trained model and selected features
with open("stacked_model.pkl", "rb") as model_file:
    stacked_model = pickle.load(model_file)

with open("selected_features.pkl", "rb") as features_file:
    selected_features = pickle.load(features_file)

# Home Page
def home():
    st.title("Credit Risk Prediction App")
    st.write("""
    Welcome to the Credit Risk Prediction App! This tool is designed to:
    - Predict the probability of loan defaults.
    - Evaluate model performance using metrics like AUC-ROC, precision, and recall.
    - Provide interpretability using SHAP values.
    
    **Steps to Use the App**:
    1. Navigate to "Prediction" to upload your dataset and get predictions.
    2. Explore "Evaluation Metrics" to view model performance.
    3. Visit "SHAP Interpretability" to understand model decisions.
    """)

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

def prediction():
    st.title("Prediction")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file for prediction", type="csv")
    if uploaded_file:
        input_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Columns (Before Processing):", input_data.columns.tolist())
        st.write("Uploaded Data Shape (Before Processing):", input_data.shape)

        # Check selected features
        st.write("Expected Features (Selected Features):", selected_features)

        # Validate uploaded data
        missing_features = [col for col in selected_features if col not in input_data.columns]
        if missing_features:
            st.error(f"The following required features are missing: {missing_features}")
            for feature in missing_features:
                input_data[feature] = 0  # Add missing features with default value

        # Reorder columns
        try:
            input_data = input_data[selected_features]
        except Exception as e:
            st.error(f"Error reordering columns: {e}")
            return

        st.write("Validated and Ordered Data Sample:")
        st.write(input_data.head())
        st.write("Final Input Data Shape (Aligned to Selected Features):", input_data.shape)

        # Preprocess non-numeric columns
        try:
            input_data['grade'] = input_data['grade'].astype(int)
            input_data['subGrade'] = input_data['subGrade'].astype(int)
            input_data['homeOwnership'] = input_data['homeOwnership'].astype(int)
            # Add more preprocessing steps for other features if necessary
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
            return

        # Apply scaling
        try:
            st.write("Shape of Data Passed to Scaler:", input_data.shape)
            scaler = StandardScaler()
            input_data_scaled = scaler.fit_transform(input_data)
            st.write("Scaled Data Shape (After Scaling):", input_data_scaled.shape)
        except Exception as e:
            st.error(f"Scaling failed: {e}")
            return

        # Predict probabilities
        try:
            predictions_proba = stacked_model.predict_proba(input_data_scaled)[:, 1]
            input_data["Default Probability"] = predictions_proba
            input_data["Prediction"] = (predictions_proba > 0.5).astype(int)
            
            st.write("Prediction Results:")
            st.write(input_data[["Default Probability", "Prediction"]])
            
            # Allow results download
            st.download_button(
                label="Download Predictions as CSV",
                data=input_data.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except ValueError as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.write("Please upload a file to proceed.")


# Evaluation Metrics Page
def evaluation():
    st.title("Evaluation Metrics")
    st.write("""
    Evaluate the model's performance using:
    - Classification Report
    - AUC-ROC Score
    - Confusion Matrix
    """)

    # Load evaluation data
    with open("evaluation_data.pkl", "rb") as eval_file:
        evaluation_data = pickle.load(eval_file)
    
    y_test = evaluation_data["y_test"]
    y_pred = evaluation_data["y_pred"]
    y_pred_proba = evaluation_data["y_pred_proba"]

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    # AUC-ROC Score
    st.subheader("AUC-ROC Score")
    auc = roc_auc_score(y_test, y_pred_proba)
    st.write(f"AUC-ROC Score: {auc:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    st.pyplot(fig)


# SHAP Interpretability Page
def shap_interpretability():
    st.title("SHAP Interpretability")
    st.write("""
    SHAP values explain how each feature contributes to the model's predictions.
    """)

    with open("shap_data.pkl", "rb") as shap_file:
        shap_data = pickle.load(shap_file)

    X_test = shap_data["X_test"]
    explainer = shap.TreeExplainer(stacked_model)
    shap_values = explainer.shap_values(X_test)

    st.subheader("SHAP Summary Plot")
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)
    st.pyplot(bbox_inches="tight")

    st.subheader("SHAP Force Plot (First Prediction)")
    shap.initjs()
    st_shap = shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0, :])
    st.write(st_shap)

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = {
    "Home": home,
    "Prediction": prediction,
    "Evaluation Metrics": evaluation,
    "SHAP Interpretability": shap_interpretability,
}
selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()
