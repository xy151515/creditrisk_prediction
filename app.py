import streamlit as st
import joblib
import pandas as pd

# Load pre-trained models and other resources
stacked_model = joblib.load("stacked_model.pkl")
selected_features = joblib.load("selected_features.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit app title
st.title("Loan Default Prediction App")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file for prediction:", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the file into a DataFrame
        input_data = pd.read_csv(uploaded_file)

        # Ensure input_data contains required features
        missing_features = [feature for feature in selected_features if feature not in input_data.columns]
        if missing_features:
            st.error(f"Missing required features: {missing_features}")
        else:
            # Filter for selected features
            input_data = input_data[selected_features]

            # Scale the data
            input_data_scaled = scaler.transform(input_data)

            # Make predictions
            predictions = stacked_model.predict(input_data_scaled)
            prediction_probabilities = stacked_model.predict_proba(input_data_scaled)[:, 1]

            # Prepare the results
            input_data["Prediction"] = ["Default" if pred == 1 else "Non-Default" for pred in predictions]
            input_data["Probability"] = prediction_probabilities

            # Display results
            st.success("Predictions generated successfully!")
            st.write(input_data)

            # Downloadable results
            csv = input_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
