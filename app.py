import streamlit as st
import pandas as pd
import joblib
import json

def load_models():
    """Load the saved models and files."""
    stacked_model = joblib.load("stacked_model_patched.pkl")  # Use the patched model
    selected_features = joblib.load("selected_features.pkl")
    try:
        scaler = joblib.load("scaler.pkl")
        if scaler is None:
            raise ValueError("Scaler object is None.")
    except Exception as e:
        raise ValueError(f"Error loading scaler: {e}")

    with open("evaluation_metrics.json", "r") as f:
        evaluation_metrics = json.load(f)
    return stacked_model, selected_features, scaler, evaluation_metrics


def predict_default(input_data, stacked_model, selected_features, scaler):
    """Predict loan default using the stacked model."""
    input_data = pd.DataFrame(input_data, columns=selected_features)
    try:
        scaled_data = scaler.transform(input_data)
    except AttributeError as e:
        raise ValueError("Scaler is not loaded correctly. Please check the scaler file.") from e

    predictions = stacked_model.predict(scaled_data)
    probabilities = stacked_model.predict_proba(scaled_data)[:, 1]
    return predictions, probabilities


# Load models and data
try:
    stacked_model, selected_features, scaler, evaluation_metrics = load_models()
except Exception as e:
    st.error("Error loading models or associated files. Ensure all necessary files are in the correct directory and not corrupted.")
    raise e

# Streamlit UI
def main():
    st.set_page_config(page_title="Loan Default Prediction", layout="wide")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox(
        "Choose an option:",
        ["Home", "Prediction", "Performance Metrics"]
    )

    if choice == "Home":
        st.title("Credit Risk Prediction App")
        st.write("Welcome to the Credit Risk Prediction App. Use this application to predict the likelihood of loan defaults based on input data.")
        st.write("Navigate through the sidebar to explore features such as predictions and model performance metrics.")

    elif choice == "Prediction":
        st.title("Loan Default Prediction")

        # File uploader for input data
        uploaded_file = st.file_uploader("Upload input file (CSV format with selected features)", type="csv")

        if uploaded_file is not None:
            try:
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
            except Exception as e:
                st.error(f"An error occurred while processing the uploaded file: {e}")

    elif choice == "Performance Metrics":
        st.title("Model Performance Metrics")

        try:
            st.write("### Accuracy")
            st.write(f"{evaluation_metrics['classification_report']['accuracy']:.4f}")

            st.write("### Precision")
            st.write(f"{evaluation_metrics['classification_report']['1']['precision']:.4f}")

            st.write("### Recall")
            st.write(f"{evaluation_metrics['classification_report']['1']['recall']:.4f}")

            st.write("### AUC-ROC")
            st.write(f"{evaluation_metrics['auc_roc_score']:.4f}")

            st.write("For detailed insights, refer to the model evaluation JSON file.")
        except KeyError as e:
            st.error(f"Missing or invalid evaluation metrics data: {e}")

if __name__ == "__main__":
    main()
