a) Problem Statement

Heart disease is one of the leading causes of death worldwide. Early prediction using machine learning can help doctors identify high-risk patients and take preventive action.

The goal of this project is to build and compare multiple machine learning classification models to predict whether a patient has heart disease based on clinical parameters. The models are evaluated using standard performance metrics and deployed using a Streamlit web application.

b) Dataset Description

Dataset Name: Heart Disease Dataset

Source: UCI / Kaggle public repository

Type: Binary Classification

Number of Features: 13

Number of Instances: 500+

Target Variable: Presence of Heart Disease (1 = Disease, 0 = No Disease)

Features Used

Age

Sex

Chest pain type (cp)

Resting blood pressure (trestbps)

Cholesterol (chol)

Fasting blood sugar (fbs)

Rest ECG (restecg)

Max heart rate (thalach)

Exercise induced angina (exang)

Oldpeak

Slope

Number of major vessels (ca)

Thal

c) Models Used and Evaluation Metrics

The following machine learning models were implemented and evaluated on the same dataset:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbor (KNN)

Naive Bayes

Random Forest (Ensemble)

XGBoost (Ensemble)

Evaluation Metrics Used

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

                 Model  Accuracy       AUC  Precision    Recall  F1 Score  \
0  Logistic Regression  0.809756  0.929810   0.761905  0.914286  0.831169   
1        Decision Tree  0.985366  0.985714   1.000000  0.971429  0.985507   
2                  KNN  0.863415  0.962905   0.873786  0.857143  0.865385   
3          Naive Bayes  0.829268  0.904286   0.807018  0.876190  0.840183   
4        Random Forest  1.000000  1.000000   1.000000  1.000000  1.000000   
5              XGBoost  1.000000  1.000000   1.000000  1.000000  1.000000   

        MCC  
0  0.630908  
1  0.971151  
2  0.726935  
3  0.660163  
4  1.000000  
5  1.000000 

Observations on Model Performance
ML Model Name	Observation about model performance
Logistic Regression	Performs well for linear relationships and gives good recall but slightly lower accuracy.
Decision Tree	Very high accuracy and precision; may overfit on training data.
KNN	Balanced performance with good recall and precision but sensitive to k value.
Naive Bayes	Fast and simple; performs moderately well but assumes feature independence.
Random Forest	Best overall performance due to ensemble learning and reduced overfitting.
XGBoost	Achieved highest performance; handles complex patterns and interactions effectively.
Streamlit Web Application Features

Dataset upload option (CSV test data)

Model selection dropdown

Evaluation metrics display

Confusion matrix / classification report

Real-time heart disease prediction interface
