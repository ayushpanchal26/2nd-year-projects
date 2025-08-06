import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open("loan.pkl", "rb"))

st.set_page_config(page_title="Loan Approval Predictor")
st.title("🏦 Loan Approval Prediction App")

st.markdown("Enter applicant details below to check if the loan will be approved.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in ₹1000s)", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", ["1.0", "0.0"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert to numeric format as model expects
input_data = pd.DataFrame({
    "Gender": [1 if gender == "Male" else 0],
    "Married": [1 if married == "Yes" else 0],
    "Dependents": [3 if dependents == "3+" else int(dependents)],
    "Education": [1 if education == "Graduate" else 0],
    "Self_Employed": [1 if self_employed == "Yes" else 0],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_amount_term],
    "Credit_History": [float(credit_history)],
    "Property_Area": [2 if property_area == "Urban" else 1 if property_area == "Semiurban" else 0]
})

# Predict
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data)[0]
    result = "✅ Loan Approved" if prediction == "Y" else "❌ Loan Rejected"
    st.success(result)
