# Heart Disease Prediction using Machine Learning

## a. Problem Statement
Heart disease is one of the leading causes of death worldwide. Early prediction can help healthcare professionals take preventive measures and provide timely treatment.  
The objective of this project is to build and compare multiple machine learning models to predict the presence of heart disease using patient health parameters.

---

## b. Dataset Description
The dataset used is the Heart Disease dataset which contains various medical attributes of patients used for predicting heart disease.

### Features:
- Age
- Sex
- Chest pain type (cp)
- Resting blood pressure (trestbps)
- Cholesterol (chol)
- Fasting blood sugar (fbs)
- Resting ECG (restecg)
- Maximum heart rate achieved (thalach)
- Exercise induced angina (exang)
- Oldpeak
- Slope
- Number of major vessels (ca)
- Thal

### Target Variable:
- 0 → No heart disease
- 1 → Presence of heart disease

---

## c. Models Used

The following machine learning models were implemented and evaluated:

- Logistic Regression  
- Decision Tree Classifier  
- k-Nearest Neighbor (kNN)  
- Naive Bayes  
- Random Forest (Ensemble Model)  
- XGBoost (Ensemble Model)  

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression |      |      |      |      |      |      |
| Decision Tree |      |      |      |      |      |      |
| kNN |      |      |      |      |      |      |
| Naive Bayes |      |      |      |      |      |      |
| Random Forest (Ensemble) |      |      |      |      |      |      |
| XGBoost (Ensemble) |      |      |      |      |      |      |

*(Fill the above values from your model evaluation results.)*

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Performs well for linear relationships and provides interpretable results. |
| Decision Tree | Easy to interpret but may overfit the training data. |
| kNN | Performance depends on the choice of K and scaling of features. |
| Naive Bayes | Fast and efficient but assumes feature independence. |
| Random Forest (Ensemble) | Reduces overfitting and improves accuracy using multiple trees. |
| XGBoost (Ensemble) | Provides best performance by capturing complex patterns and interactions. |

---

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis
4. Model Training
5. Model Evaluation using:
   - Accuracy
   - AUC
   - Precision
   - Recall
   - F1 Score
   - MCC
6. Model Comparison
7. Deployment using Streamlit

---

## Tools & Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib / Seaborn
