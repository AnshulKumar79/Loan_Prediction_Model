import streamlit as st
import pandas as pd
import joblib


st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Loan Approval Predictor")
st.write("""
Fill in the applicant's details below to predict whether their loan will be approved or rejected.
""")

st.divider()


@st.cache_resource
def load_model():
    try:
        # Replace 'model.pkl' with the actual path to your saved model
        model = joblib.load("loan_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file 'loan_model.pkl' not found. Please ensure it is in the same directory.")
        return None

model = load_model()


st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    income = st.number_input("Annual Income (₹)", min_value=0.0, value=50000.0, step=1000.0)

with col2:
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700, step=1)
    loan_amt = st.number_input("Requested Loan Amount (₹)", min_value=1000.0, value=200000.0, step=5000.0)


st.divider()

if st.button("Predict Loan Status", type="primary"):
    if model is not None:
        # Create a DataFrame for the inputs (matches typical scikit-learn input formats)
        # Ensure these column names match exactly what your model was trained on!
        input_data = pd.DataFrame({
            'Age': [age],
            'Income': [income],
            'LoanAmount': [loan_amt],
            'CreditScore': [cibil_score]
        })
        
        # Make the prediction
        try:
            prediction = model.predict(input_data)[0]
            
            # Assuming your model outputs 1 for Approved and 0 for Rejected
            if prediction == 1:
                st.success("✅ **Prediction: Loan Approved!**")
                st.balloons()
            else:
                st.error("❌ **Prediction: Loan Rejected.**")
                
        except Exception as e:
            st.error(f"An error occurred during prediction. Please check if the input features match your model's expected features. Error: {e}")
    else:
        st.warning("Cannot make a prediction because the model is not loaded.")