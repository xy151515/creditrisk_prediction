import streamlit as st
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to clean column names
def clean_column_names(df):
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
    return df

# Function to load the pre-trained model, scaler, and metrics
@st.cache_data
def load_model_scaler_and_metrics():
    with open('stacked_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('evaluation_metrics.json', 'r') as file:
        metrics = json.load(file)
    with open('selected_features.pkl', 'rb') as file:
        selected_features = pickle.load(file)
    return model, scaler, metrics, selected_features

model, scaler, metrics, selected_features = load_model_scaler_and_metrics()

st.title("Fraud Detection in Electricity and Gas Consumption")
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Go to:", ["Home", "Training Data", "Evaluation Metrics", "Prediction"])

if menu == "Home":
    st.write("""
    ## Welcome to the Fraud Detection App
    This app uses a LightGBM model to predict fraud in electricity and gas consumption based on provided data.
    Use the navigation menu to explore training data, evaluation metrics, or make predictions with your test data.
    
    ### Steps:
    1. Review the training data.
    2. Check the evaluation metrics of the trained model.
    3. Upload your test data for fraud predictions.
    """)

elif menu == "Training Data":
    st.subheader("Training Data")
    train_file_path = 'train_sample.csv'
    model_train = pd.read_csv(train_file_path)
    st.write("### Training Data Preview")
    st.dataframe(model_train.head())
    st.write("### Columns in Training Data")
    st.write(list(model_train.columns))

elif menu == "Evaluation Metrics":
    st.subheader("Model Evaluation Metrics")
    evaluation_metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1-Score']]
    })
    st.write(evaluation_metrics)

    # Feature importance
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': selected_features, 'importance': importance})
    feature_importance = feature_importance.sort_values(by='importance', ascending=False).head(10)  # Show top 10 features

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
    ax.set_title("Top 10 Feature Importance")
    st.pyplot(fig)

elif menu == "Prediction":
    st.subheader("Upload Test Data for Predictions")

    # File uploader for testing data
    uploaded_test_file = st.file_uploader("Choose a CSV file for prediction", type="csv")
    if uploaded_test_file is not None:
        test_data = pd.read_csv(uploaded_test_file)
        st.write("### Test Data Preview:", test_data.head())

        # Ensure the categorical columns exist in the test DataFrame
        categorical_columns = ['disrict', 'client_catg', 'region', 'tarif_type_elec_mode', 'tarif_type_gaz_mode']
        existing_test_categorical_columns = [col for col in categorical_columns if col in test_data.columns]

        # Apply the same preprocessing steps to the test data
        test_data_encoded = pd.get_dummies(test_data, columns=existing_test_categorical_columns, drop_first=True)
        test_data_encoded = clean_column_names(test_data_encoded)
        test_data_encoded = test_data_encoded.reindex(columns=selected_features, fill_value=0)  # Align columns with training

        numerical_features = [col for col in selected_features if col in test_data_encoded.columns and test_data_encoded[col].dtype in ['int64', 'float64']]
        test_data_encoded[numerical_features] = scaler.transform(test_data_encoded[numerical_features])

        # Select the same features as used in training
        X_test_final = test_data_encoded[selected_features]

        # Make predictions
        y_pred_prob = model.predict_proba(X_test_final)[:, 1]
        y_pred = (y_pred_prob >= 0.6).astype(int)  # Threshold set to 0.60 for fraud

        # Create a DataFrame with client_id, prediction probabilities, and predicted values
        results = pd.DataFrame({
            'client_id': test_data['client_id'],
            'fraud_probability': y_pred_prob,
            'predicted_value': y_pred
        })

        st.write("### Prediction Results")
        st.dataframe(results)
        st.download_button(
            label="Download Predictions",
            data=results.to_csv(index=False),
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

        # Visualization of prediction results
        st.subheader("Prediction Results Visualization")
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        results['predicted_value'].value_counts().plot(kind='bar', ax=ax[0])
        ax[0].set_title("Count of Predicted Fraudulent and Non-Fraudulent Transactions")
        ax[0].set_xlabel("Predicted Value (1=Fraud, 0=Non-Fraud)")
        ax[0].set_ylabel("Count")

        sns.histplot(results['fraud_probability'], bins=20, kde=True, ax=ax[1])
        ax[1].set_title("Distribution of Fraud Probability Scores")
        ax[1].set_xlabel("Fraud Probability")
        ax[1].set_ylabel("Frequency")

        st.pyplot(fig)
