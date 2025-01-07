import streamlit as st
import pandas as pd
import joblib
import json

def load_models():
    """Load the saved models and files."""
    stacked_model = joblib.load("stacked_model.pkl")
    selected_features = joblib.load("selected_features.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("evaluation_metrics.json", "r") as f:
        evaluation_metrics = json.load(f)
    return stacked_model, selected_features, scaler, evaluation_metrics

def predict_default(input_data, stacked_model, selected_features, scaler):
    """Predict loan default using the stacked model."""
    input_data = pd.DataFrame(input_data, columns=selected_features)
    scaled_data = scaler.transform(input_data)
    predictions = stacked_model.predict(scaled_data)
    probabilities = stacked_model.predict_proba(scaled_data)[:, 1]
    return predictions, probabilities

# Load models and data
stacked_model, selected_features, scaler, evaluation_metrics = load_models()

# Streamlit UI
# App title
st.title("Credit Risk Prediction App")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose an option:",
    ["Home", "Upload Data", "Performance Metrics"]
)

# Home section
if app_mode == "Home":
    st.header("Welcome to the Credit Risk Prediction App")
    st.write("""
        This application predicts the likelihood of loan default using a pre-trained machine learning model.
        Features include:
        - Uploading new loan application data for prediction
        - Viewing model performance metrics
        - Understanding feature importance
    """)

    elif choice == "Prediction":
        st.title("Loan Default Prediction")

        # File uploader for input data
        uploaded_file = st.file_uploader("Upload input file (CSV format with selected features)", type="csv")

        if uploaded_file is not None:
            input_data = pd.read_csv(uploaded_file)

            # Check if required features are in the input data
            missing_features = [feature for feature in selected_features if feature not in input_data.columns]
            if missing_features:
                st.error(f"Missing required features: {missing_features}")
            else:
                predictions, probabilities = predict_default(input_data[selected_features], stacked_model, selected_features, scaler)
                input_data["Prediction"] = predictions
                input_data["Default Probability"] = probabilities

                # Display predictions
                st.write("Predictions:")
                st.dataframe(input_data)

                # Option to download predictions
                csv = input_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

# Performance Metrics section
    elif app_mode == "Performance Metrics":
        st.header("Model Performance Metrics")
        st.metric("Accuracy", f"{evaluation_metrics['accuracy']:.4f}")
        st.metric("Precision", f"{evaluation_metrics['precision']:.4f}")
        st.metric("Recall", f"{evaluation_metrics['recall']:.4f}")
        st.metric("AUC-ROC", f"{evaluation_metrics['auc_roc']:.4f}")
    
        st.write("For detailed insights, refer to the model evaluation JSON file.")

if __name__ == "__main__":
    main()
