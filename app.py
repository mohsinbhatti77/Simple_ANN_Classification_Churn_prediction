import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# -------------------------
# Load saved components
# -------------------------
model = load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('one_hot_encoder_geography.pkl', 'rb') as f:
    one_hot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

expected_columns = scaler.feature_names_in_  # Extracted from scaler during training

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Churn Prediction App", layout="centered")

st.title("🔍 Customer Churn Prediction")
st.markdown("Enter customer details to predict churn probability.")

# Input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=1000, value=650)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Account Balance", value=100000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"])
is_active = st.radio("Is Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", value=50000.0)

# -------------------------
# Preprocess Input
# -------------------------
if st.button("Predict Churn"):

    # Encode categorical
    gender_encoded = label_encoder_gender.transform([gender])[0]
    geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

    # Base input
    input_data = pd.DataFrame([{
        'CreditScore': credit_score,
        'Gender': gender_encoded,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': 1 if has_cr_card == 'Yes' else 0,
        'IsActiveMember': 1 if is_active == 'Yes' else 0,
        'EstimatedSalary': estimated_salary
    }])

    # Merge encoded geo
    input_data = pd.concat([input_data, geo_df], axis=1)

    # Align with scaler columns
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Fill missing columns with 0

    # Reorder columns
    input_data = input_data[list(expected_columns)]

    # Scale input
    input_scaled = scaler.transform(input_data)

 # Make prediction
    probability = model.predict(input_scaled)[0][0]
    churned = int(probability > 0.5)  # 1 = churn, 0 = stay

    # Output the result (clean and clear)
    st.subheader("📈 Prediction Result")
    if churned:
        st.error(f"🔴 The customer is likely to churn. (Probability: {probability:.2f})")
    else:
        st.success(f"🟢 The customer is likely to stay. (Probability: {probability:.2f})")

