import streamlit as st
import pickle
import numpy as np

# Load model & scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Title
st.set_page_config(page_title="AI Churn Predictor", layout="centered")
st.title("🚀 AI Customer Churn Analyzer")

st.markdown("Fill the details below to predict customer churn risk")

# Inputs
creditscore = st.number_input("Credit Score", min_value=300, max_value=900)
age = st.number_input("Age", min_value=18, max_value=100)
balance = st.number_input("Balance")
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10)
salary = st.number_input("Estimated Salary")

# Button
if st.button("Analyze Risk"):

    # Prepare data
    data = np.array([[creditscore, age, balance, tenure, salary]])
    data = scaler.transform(data)

    # Prediction
    prediction = model.predict(data)
    prob = model.predict_proba(data)

    probability = round(prob[0][1] * 100, 2)
    retention = round(100 - probability, 2)

    # Output
    st.subheader("📊 Result")

    col1, col2 = st.columns(2)

    col1.metric("⚠️ Churn Risk", f"{probability}%")
    col2.metric("✅ Retention", f"{retention}%")

    # Decision
    if probability < 30:
        st.success("✅ LOW RISK")
    elif probability < 70:
        st.warning("⚠️ MEDIUM RISK")
    else:
        st.error("❌ HIGH RISK")

    # Progress bar
    st.progress(int(probability))

    # Chart
    st.subheader("📊 Risk Distribution")
    st.bar_chart({
        "Churn": [probability],
        "Retention": [retention]
    })