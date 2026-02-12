import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Heart Disease ML App", layout="wide")

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
    st.subheader("Dataset Preview")
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

            # AUC calculation
            try:
                prob = model.predict_proba(X)[:, 1]
                auc = roc_auc_score(y, prob)
            except:
                auc = "Not available"

            acc = accuracy_score(y, predictions)
            precision = precision_score(y, predictions)
            recall = recall_score(y, predictions)
            f1 = f1_score(y, predictions)
            mcc = matthews_corrcoef(y, predictions)
            cm = confusion_matrix(y, predictions)

            # -------------------------------
            # Metrics table
            # -------------------------------
            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
                "Value": [acc, auc, precision, recall, f1, mcc]
            })

            st.subheader("📊 Evaluation Metrics Table")
            st.dataframe(metrics_df, use_container_width=True)

            # -------------------------------
            # Confusion Matrix Heatmap
            # -------------------------------
            st.subheader("📉 Confusion Matrix")

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
            st.pyplot(fig)

            # -------------------------------
            # Classification Report table
            # -------------------------------
            report_dict = classification_report(y, predictions, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()

            st.subheader("📄 Classification Report Table")
            st.dataframe(report_df, use_container_width=True)

        # -------------------------------
        # Prediction section
        # -------------------------------
        st.subheader("🔎 Make Prediction")

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
