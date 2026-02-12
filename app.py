import streamlit as st
import pandas as pd
import joblib

st.title("Heart Disease Prediction Web App")

st.write("Enter patient details to predict heart disease")

# Load trained models
dt_model = joblib.load("model/decision_tree_model.pkl")
knn_model = joblib.load("model/knn_model.pkl")
nb_model = joblib.load("model/naive_bayes_model.pkl")
rf_model = joblib.load("model/random_forest_model.pkl")
xgb_model = joblib.load("model/xgboost_model.pkl")

# Input fields
age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.number_input("Chest Pain Type (0–3)", 0, 3)
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = True, 0 = False)", [1, 0])
restecg = st.number_input("Resting ECG (0–2)", 0, 2)
thalach = st.number_input("Max Heart Rate", 60, 220)
exang = st.selectbox("Exercise Induced Angina", [1, 0])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0)
slope = st.number_input("Slope", 0, 2)
ca = st.number_input("Number of Major Vessels", 0, 4)
thal = st.number_input("Thal (0–3)", 0, 3)

# Create input dataframe
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Prediction
if st.button("Predict"):

    dt_pred = dt_model.predict(input_data)
    knn_pred = knn_model.predict(input_data)
    nb_pred = nb_model.predict(input_data)
    rf_pred = rf_model.predict(input_data)
    xgb_pred = xgb_model.predict(input_data)

    st.subheader("Predictions from Models")

    st.write("Decision Tree:", "Disease" if dt_pred[0] == 1 else "No Disease")
    st.write("KNN:", "Disease" if knn_pred[0] == 1 else "No Disease")
    st.write("Naive Bayes:", "Disease" if nb_pred[0] == 1 else "No Disease")
    st.write("Random Forest:", "Disease" if rf_pred[0] == 1 else "No Disease")
    st.write("XGBoost:", "Disease" if xgb_pred[0] == 1 else "No Disease")

