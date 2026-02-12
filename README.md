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

Model Performance Comparison Table
ML Model Name	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.8098	0.9298	0.7619	0.9143	0.8312	0.63
Decision Tree	0.9854	0.9857	1.00	0.9714	0.9855	0.9711
KNN	0.8634	0.9629	0.8738	0.8571	0.8654	0.7269
Naive Bayes	0.8293	0.9043	0.8070	0.8762	0.8402	0.6602
Random Forest	1.00	1.00	1.00	1.00	1.00	1.00
XGBoost	1.00	1.00	1.00	1.00	1.00	1.00
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
