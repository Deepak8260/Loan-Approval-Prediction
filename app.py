import streamlit as st
import pandas as pd
import numpy as np
import pickle

def load_model_resources():
    """Load the trained model and preprocessing tools from a pickle file."""
    with open("model.pkl", "rb") as file:
        resources = pickle.load(file)
    return resources["model"], resources["scaler"], resources["encoder"], resources["label_binarizer"]

def get_user_input():
    """Capture user input for loan prediction."""
    st.markdown("<h3>Enter Loan Applicant Details</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        current_loan_amount = st.number_input("Current Loan Amount", value=10000, step=100, min_value=1000)
        term = st.selectbox("Term", ['Short Term', 'Long Term'])
        credit_score = st.number_input("Credit Score", value=700, step=1, min_value=300, max_value=850)
        years_in_job = st.number_input("Years in Current Job", value=5, step=1, min_value=0)
        home_ownership = st.selectbox("Home Ownership", ['Own', 'Mortgage', 'Rent', 'Other'])
        annual_income = st.number_input("Annual Income", value=50000, step=1000, min_value=1000)
        purpose = st.selectbox("Purpose", ['Debt Consolidation', 'Home Improvements', 'Other'])
        monthly_debt = st.number_input("Monthly Debt", value=500, step=10, min_value=0)
    
    with col2:
        years_credit_history = st.number_input("Years of Credit History", value=5, step=1, min_value=0)
        months_since_last_delinquent = st.number_input("Months Since Last Delinquent", value=12, step=1, min_value=0)
        num_open_accounts = st.number_input("Number of Open Accounts", value=5, step=1, min_value=0)
        num_credit_problems = st.number_input("Number of Credit Problems", value=0, step=1, min_value=0)
        current_credit_balance = st.number_input("Current Credit Balance", value=5000, step=100, min_value=0)
        max_open_credit = st.number_input("Maximum Open Credit", value=15000, step=500, min_value=0)
        bankruptcies = st.number_input("Bankruptcies", value=0, step=1, min_value=0)
        tax_liens = st.number_input("Tax Liens", value=0, step=1, min_value=0)
    
    return {
        "Current Loan Amount": current_loan_amount,
        "Term": term,
        "Credit Score": credit_score,
        "Years in Current Job": years_in_job,
        "Home Ownership": home_ownership,
        "Annual Income": annual_income,
        "Purpose": purpose,
        "Monthly Debt": monthly_debt,
        "Years of Credit History": years_credit_history,
        "Months Since Last Delinquent": months_since_last_delinquent,
        "Number of Open Accounts": num_open_accounts,
        "Number of Credit Problems": num_credit_problems,
        "Current Credit Balance": current_credit_balance,
        "Maximum Open Credit": max_open_credit,
        "Bankruptcies": bankruptcies,
        "Tax Liens": tax_liens
    }

def preprocess_input(user_input, encoder, scaler):
    """Preprocess user input by applying one-hot encoding and scaling."""
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=encoder, fill_value=0)  
    return scaler.transform(input_encoded)

def predict_loan_status(model, scaler, encoder, label_binarizer, user_input):
    """Predict loan approval status based on user input."""
    input_scaled = preprocess_input(user_input, encoder, scaler)
    prediction = model.predict(input_scaled)
    return label_binarizer.inverse_transform(np.array(prediction).reshape(-1, 1))[0]

def main():
    """Streamlit application entry point."""
    st.markdown("<h1>📊 Loan Approval Prediction</h1>", unsafe_allow_html=True)
    
    model, scaler, encoder, label_binarizer = load_model_resources()
    user_input = get_user_input()
    
    if st.button("🚀 Predict Loan Status"):
        predicted_label = predict_loan_status(model, scaler, encoder, label_binarizer, user_input)
        st.write(f"Loan Approval Prediction: **{predicted_label}**")

if __name__ == "__main__":
    main()
