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
st.title("Customer Churn Prediction App")
st.markdown("Welcome! Upload a dataset and predict churn using a pre-trained model.")

# Sidebar
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load pre-trained model
model = joblib.load("churn_model.pkl")

# Initialize encoder
encoder = LabelEncoder()

# Main logic
if uploaded_file is not None:
    try:
        # Try reading with UTF-8 encoding
        try:
            data = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            data = pd.read_csv(uploaded_file, encoding='latin1')

        st.subheader("Uploaded Data")
        st.write(data)

        # Encode categorical columns
        data_processed = data.copy()
        for col in data_processed.select_dtypes(include='object').columns:
            data_processed[col] = encoder.fit_transform(data_processed[col].astype(str))

        st.subheader("Preprocessed Data")
        st.write(data_processed)

        # Prediction (assuming model.predict works on the whole DataFrame)
        predictions = model.predict(data_processed)

        # Add prediction to output
        data['Churn_Prediction'] = predictions
        st.subheader("Predictions")
        st.write(data)

        # Option to download results
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions as CSV", data=csv, file_name="churn_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
