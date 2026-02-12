import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("❤️ Heart Disease Prediction ML App")

# -------------------------------
# Load trained models (.pkl files)
# -------------------------------
try:
    lr_model = joblib.load("model/logistic_regression_model.pkl")
    dt_model = joblib.load("model/decision_tree_model.pkl")
    knn_model = joblib.load("model/knn_model.pkl")
    nb_model = joblib.load("model/naive_bayes_model.pkl")
    rf_model = joblib.load("model/random_forest_model.pkl")
    xgb_model = joblib.load("model/xgboost_model.pkl")
except:
    st.error("Model files not found! Upload .pkl files inside model/ folder.")

# -------------------------------
# Upload dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload heart dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview")
    st.dataframe(df.head())

    if "target" not in df.columns:
        st.error("Dataset must contain 'target' column")
    else:
        X = df.drop("target", axis=1)
        y = df["target"]

        # -------------------------------
        # Model selection dropdown
        # -------------------------------
        model_choice = st.selectbox(
            "Select ML Model",
            (
                "Logistic Regression",
                "Decision Tree",
                "KNN",
                "Naive Bayes",
                "Random Forest",
                "XGBoost"
            )
        )

        if model_choice == "Logistic Regression":
            model = lr_model
        elif model_choice == "Decision Tree":
            model = dt_model
        elif model_choice == "KNN":
            model = knn_model
        elif model_choice == "Naive Bayes":
            model = nb_model
        elif model_choice == "Random Forest":
            model = rf_model
        else:
            model = xgb_model

        # -------------------------------
        # Evaluate model
        # -------------------------------
        if st.button("Evaluate Model"):
            predictions = model.predict(X)

            acc = accuracy_score(y, predictions)
            cm = confusion_matrix(y, predictions)

            st.subheader("Model Accuracy")
            st.write(acc)

            st.subheader("Confusion Matrix")
            st.write(cm)

        # -------------------------------
        # Prediction section
        # -------------------------------
        st.subheader("Make Prediction")

        input_data = []
        for col in X.columns:
            val = st.number_input(f"Enter {col}", value=0.0)
            input_data.append(val)

        if st.button("Predict Heart Disease"):
            input_df = pd.DataFrame([input_data], columns=X.columns)
            pred = model.predict(input_df)[0]

            if pred == 1:
                st.error("⚠️ Person has Heart Disease")
            else:
                st.success("✅ Person is Normal")
