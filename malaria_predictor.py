
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🦟 Malaria Prediction App")
st.write("App Developed By David Emmanuel SPD400L FUL:")
st.write("Enter patient symptoms below to predict the presence of malaria:")

fever = st.slider("Body Temperature (°C)", 35.0, 42.0, 37.0)
headache = st.selectbox("Severe Headache?", ["Yes", "No"])
vomiting = st.selectbox("Vomiting?", ["Yes", "No"])
sweating = st.selectbox("Excessive Sweating?", ["Yes", "No"])
chills = st.selectbox("Chills?", ["Yes", "No"])
muscle_pain = st.selectbox("Muscle Pain?", ["Yes", "No"])
diarrhea = st.selectbox("Diarrhea?", ["Yes", "No"])

input_data = pd.DataFrame([{
    "fever": fever,
    "headache": 1 if headache == "Yes" else 0,
    "vomiting": 1 if vomiting == "Yes" else 0,
    "sweating": 1 if sweating == "Yes" else 0,
    "chills": 1 if chills == "Yes" else 0,
    "muscle_pain": 1 if muscle_pain == "Yes" else 0,
    "diarrhea": 1 if diarrhea == "Yes" else 0
}])

scaled_input = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(scaled_input)[0]
    if prediction == 1:
        st.error("Prediction: Malaria Detected 😷")
    else:
        st.success("Prediction: No Malaria Detected ✅")
