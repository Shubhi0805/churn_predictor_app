import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Set page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide"
)

# Branding
st.markdown("<h1 style='color:#4CAF50;'>Customer Churn Prediction App</h1>", unsafe_allow_html=True)
st.markdown("Upload a dataset to predict churn using a pre-trained Random Forest model.")

# Sidebar for file upload
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load pre-trained model and encoder
model = joblib.load("churn_model.pkl")
encoder = LabelEncoder()

# Specify the features used in model training
model_features = ['gender', 'SeniorCitizen', 'Partner', 'tenure', 'MonthlyCharges', 'Contract', 'PaymentMethod']  # ✨ Change based on your model training

# Main logic
if uploaded_file is not None:
    try:
        # Read CSV file
        try:
            data = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            data = pd.read_csv(uploaded_file, encoding='latin1')

        st.subheader("Uploaded Data")
        st.write(data.head())

        # Check for required features
        missing_cols = [col for col in model_features if col not in data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            # Filter only model features
            data_model = data[model_features].copy()

            # Encode categorical columns
            for col in data_model.select_dtypes(include='object').columns:
                data_model[col] = encoder.fit_transform(data_model[col].astype(str))

            st.subheader("Preprocessed Data")
            st.write(data_model.head())

            # Make prediction
            predictions = model.predict(data_model)

            # Attach predictions to original dataset
            data['Churn_Prediction'] = predictions
            st.subheader("Predictions")
            st.write(data)

            # Download predictions
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions as CSV", data=csv, file_name="churn_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to begin.")
