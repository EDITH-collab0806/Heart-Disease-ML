# Heart Disease Prediction – End-to-End Machine Learning Project

This project builds multiple Machine Learning classification models to predict the presence of heart disease using clinical patient data.  
It includes preprocessing, model training, evaluation, comparison, and deployment using Streamlit.

---

## 🚀 Project Workflow

1. Data Collection (Heart Disease Dataset)
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Model Comparison
7. Deployment using Streamlit

---

## 🤖 Machine Learning Models Used

- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbor (KNN)  
- Naive Bayes  
- Random Forest  
- XGBoost  

---

## 📊 Model Performance Comparison

| ML Model Name        | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC     |
|----------------------|----------|---------|-----------|---------|----------|---------|
| Logistic Regression  | 0.809756 | 0.929810| 0.761905  | 0.914286| 0.831169 | 0.630908|
| Decision Tree        | 0.985366 | 0.985714| 1.000000  | 0.971429| 0.985507 | 0.971151|
| KNN                  | 0.863415 | 0.962905| 0.873786  | 0.857143| 0.865385 | 0.726935|
| Naive Bayes          | 0.829268 | 0.904286| 0.807018  | 0.876190| 0.840183 | 0.660163|
| Random Forest        | 1.000000 | 1.000000| 1.000000  | 1.000000| 1.000000 | 1.000000|
| XGBoost              | 1.000000 | 1.000000| 1.000000  | 1.000000| 1.000000 | 1.000000|

---

## 📈 Observations on Model Performance

| ML Model Name        | Observation |
|----------------------|-------------|
| Logistic Regression  | Performs well for linear relationships; good recall with moderate accuracy. |
| Decision Tree        | Very high accuracy and precision; may overfit training data. |
| KNN                  | Balanced performance; sensitive to K value and feature scaling. |
| Naive Bayes          | Fast and simple; moderate performance due to independence assumption. |
| Random Forest        | Strong performance with reduced overfitting; highly reliable. |
| XGBoost              | Best performing model with excellent prediction capability. |

---

## 📊 Evaluation Metrics Used

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- AUC Score  
- Matthews Correlation Coefficient (MCC)  
- Confusion Matrix  
- Classification Report  

---

## 🖥️ Streamlit App Features

- Select ML model  
- View model accuracy & evaluation metrics  
- Display confusion matrix  
- Display classification report  
- Predict heart disease risk for new patient data  

---

## 📂 Project Structure

├── app.py
├── heart.csv
├── models/
├── requirements.txt
└── README.md

