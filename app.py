import streamlit as st
import pandas as pd
import joblib
import json

def load_models():
    """Load and patch the saved models and files."""
    import warnings
    from sklearn.ensemble import StackingClassifier
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    try:
        stacked_model = joblib.load("stacked_model.pkl")
        
        # Patch the _label_encoder attribute if missing
        if isinstance(stacked_model, StackingClassifier) and not hasattr(stacked_model, "_label_encoder"):
            stacked_model._label_encoder = None

        selected_features = joblib.load("selected_features.pkl")
        scaler = joblib.load("scaler.pkl")
        if not isinstance(scaler, StandardScaler):
            raise ValueError("Scaler is not a valid StandardScaler instance.")
    except Exception as e:
        st.error(f"Error loading models or scaler: {e}")
        raise

    with open("evaluation_metrics.json", "r") as f:
        evaluation_metrics = json.load(f)

    return stacked_model, selected_features, scaler, evaluation_metrics


def validate_input_data(input_data, selected_features):
    """Validate the input data against selected features."""
    missing_features = [feature for feature in selected_features if feature not in input_data.columns]
    if missing_features:
        raise ValueError(f"Missing features in input data: {missing_features}")

    if input_data.isnull().values.any():
        raise ValueError("Input data contains null values. Please clean the data.")

    return input_data

def predict_default(input_data, stacked_model, selected_features, scaler):
    """Predict loan default using the stacked model."""
    input_data = pd.DataFrame(input_data, columns=selected_features)

    # Ensure the scaler is valid
    if scaler is None or not hasattr(scaler, "transform"):
        raise ValueError("Scaler is not initialized or improperly loaded.")

    try:
        scaled_data = scaler.transform(input_data)
    except Exception as e:
        raise ValueError(f"Error during scaling: {e}")

    try:
        predictions = stacked_model.predict(scaled_data)
        probabilities = stacked_model.predict_proba(scaled_data)[:, 1]
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")

    return predictions, probabilities


# Load models and data
try:
    stacked_model, selected_features, scaler, evaluation_metrics = load_models()
except Exception as e:
    st.error("Failed to load models or associated files. Please check the uploaded model files.")
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
                validate_input_data(input_data, selected_features)

                predictions, probabilities = predict_default(input_data[selected_features], stacked_model, selected_features, scaler)
                input_data["Prediction"] = predictions
                input_data["Default Probability"] = probabilities

                # Display predictions
                st.write("Predictions:")
                st.dataframe(input_data)

                # Option to download predictions
                csv = input_data.to_csv(index=False).encode("utf-8")
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
