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
def main():
    st.set_page_config(page_title="Loan Default Prediction", layout="wide")

    # Sidebar navigation
    menu = ["Home", "Prediction", "Evaluation Metrics"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Home":
        st.title("Welcome to the Loan Default Prediction App")
        st.write("Use this app to predict whether a loan will default based on input data.")
        st.write("Navigate to the Prediction section to make predictions or to the Evaluation Metrics section to view model performance.")

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

    elif choice == "Evaluation Metrics":
        st.title("Evaluation Metrics")

        st.write("### Classification Report")
        st.json(evaluation_metrics["classification_report"])

        st.write("### AUC-ROC Score")
        st.write(f"{evaluation_metrics['auc_roc_score']:.4f}")

if __name__ == "__main__":
    main()
