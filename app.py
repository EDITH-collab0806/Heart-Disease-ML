import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.title("Heart Disease Prediction Web App")

st.write("Upload test dataset OR enter patient details manually")

# Load trained models
dt_model = joblib.load("model/decision_tree_model.pkl")
knn_model = joblib.load("model/knn_model.pkl")
nb_model = joblib.load("model/naive_bayes_model.pkl")
rf_model = joblib.load("model/random_forest_model.pkl")
xgb_model = joblib.load("model/xgboost_model.pkl")

models = {
    "Decision Tree": dt_model,
    "KNN": knn_model,
    "Naive Bayes": nb_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

# -------------------------------
# (a) DATASET UPLOAD OPTION
# -------------------------------
st.header("Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# -------------------------------
# (b) MODEL SELECTION DROPDOWN
# -------------------------------
model_name = st.selectbox("Select Model", list(models.keys()))
selected_model = models[model_name]

# -------------------------------
# PREDICTION USING CSV
# -------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset Preview", data.head())

    if st.button("Run Prediction on Dataset"):
        X = data.drop("target", axis=1)
        y = data["target"]

        y_pred = selected_model.predict(X)

        # -------------------------------
        # (c) EVALUATION METRICS
        # -------------------------------
        st.subheader("Evaluation Metrics")

        acc = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        st.write("Accuracy:", acc)
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("F1 Score:", f1)

        # -------------------------------
        # (d) CONFUSION MATRIX
        # -------------------------------
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        st.write(cm)

        st.subheader("Classification Report")
        report = classification_report(y, y_pred)
        st.text(report)

# -------------------------------
# MANUAL INPUT SECTION
# -------------------------------
st.header("Manual Patient Input")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.number_input("Chest Pain Type (0–3)", 0, 3)
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120", [1, 0])
restecg = st.number_input("Resting ECG (0–2)", 0, 2)
thalach = st.number_input("Max Heart Rate", 60, 220)
exang = st.selectbox("Exercise Induced Angina", [1, 0])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0)
slope = st.number_input("Slope", 0, 2)
ca = st.number_input("Number of Major Vessels", 0, 4)
thal = st.number_input("Thal (0–3)", 0, 3)

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

if st.button("Predict (Manual Input)"):
    pred = selected_model.predict(input_data)

    st.subheader("Prediction Result")
    st.success("Heart Disease Detected" if pred[0] == 1 else "No Heart Disease")
