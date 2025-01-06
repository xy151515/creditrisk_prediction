import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and feature list
with open("stacked_model.pkl", "rb") as model_file:
    stacked_model = pickle.load(model_file)

with open("selected_features.pkl", "rb") as features_file:
    selected_features = pickle.load(features_file)

# Title of the Streamlit app
st.title("Credit Risk Prediction App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type="csv")

if uploaded_file:
    # Read the uploaded CSV file
    input_data = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.write(input_data.head())
    
    # Ensure required features are present
    missing_features = [col for col in selected_features if col not in input_data.columns]
    if missing_features:
        st.error(f"The following required features are missing: {missing_features}")
    else:
        # Select required features and standardize
        input_data = input_data[selected_features]
        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(input_data)
        
        # Predict probabilities
        predictions_proba = stacked_model.predict_proba(input_data_scaled)[:, 1]
        
        # Assign predictions
        input_data["Default Probability"] = predictions_proba
        input_data["Prediction"] = (predictions_proba > 0.5).astype(int)
        
        # Display results
        st.write("Prediction Results:")
        st.write(input_data[["Default Probability", "Prediction"]])
        
        # Option to download results
        st.download_button(
            label="Download Predictions as CSV",
            data=input_data.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv",
        )
