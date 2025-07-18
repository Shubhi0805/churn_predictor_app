# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 10:12:27 2025

@author: Shubhi Sharma
"""

# churn_app.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ==============================
# Load logo (Optional)
# ==============================
try:
    logo = Image.open("logo.png")
    st.image(logo, width=120)
except:
    pass  # If logo not found, continue

# ==============================
# App Title and Description
# ==============================
st.title("E-Commerce Customer Churn Prediction App")

st.markdown("""
This application helps businesses understand and predict customer churn.  
You can explore data trends, visualize key patterns, and predict whether a specific customer is likely to leave the platform based on their activity.

---
""")

# ==============================
# About the App and Model Info
# ==============================
st.subheader("About the App")
st.write("""
This tool is designed for e-commerce platforms to identify which customers might stop using the service.  
The prediction is based on behavioral features such as order activity, session time, complaints, and more.
""")

st.subheader("Features Used")
st.markdown("""
- Tenure (how long the customer has been active)
- Total number of orders
- Average session time
- Number of complaint tickets
- Coupon usage
- Returned orders
- Last visit (days ago)
""")

st.subheader("Model Details")
st.markdown("""
- **Algorithm Used:** Random Forest Classifier  
- **Scaler:** StandardScaler  
- **Model Accuracy:** Around 88% on the test set  
- The model has been trained on cleaned and preprocessed historical customer data.
""")

st.markdown("---")

# ==============================
# File uploader for EDA
# ==============================
st.sidebar.header("Upload Customer CSV for EDA")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # EDA Section
    st.subheader("Exploratory Data Analysis")

    st.write("Sample Data")
    st.dataframe(data.head())

    st.write("Churn Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=data, x='Churned', ax=ax1)
    st.pyplot(fig1)

    st.write("Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# ==============================
# Prediction Section
# ==============================
st.subheader("Predict Individual Customer Churn")

cust_id = st.text_input("Customer ID (e.g., CUST0030)")
tenure = st.number_input("Tenure (in months)", min_value=0)
order_count = st.number_input("Total Orders", min_value=0)
avg_session = st.number_input("Average Session Time (in minutes)", min_value=0.0)
complaints = st.number_input("Number of Complaint Tickets", min_value=0)
used_coupon = st.number_input("Used Coupon? (0 for No, 1 for Yes)", min_value=0, max_value=1)
returned_orders = st.number_input("Returned Orders", min_value=0)
last_visited = st.number_input("Last Visited (days ago)", min_value=0)

# Load trained model and scaler
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

if st.button("Predict Churn"):
    try:
        input_data = np.array([[tenure, order_count, avg_session, complaints,
                                used_coupon, returned_orders, last_visited]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.error(f"Customer {cust_id} is likely to churn.")
        else:
            st.success(f"Customer {cust_id} is likely to stay.")
    except Exception as e:
        st.warning("An error occurred while predicting.")
        st.text(str(e))

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown(
    "<center><sub>Built by <b>Shubhi Sharma</b> | BIAS, Bhimtal</sub></center>",
    unsafe_allow_html=True
)
