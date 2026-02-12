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

## 📈 Detailed Observations on Model Performance

| ML Model Name | Observation |
|--------------|-------------|
| Logistic Regression | Works well when the relationship between features and target is linear. Shows strong recall, meaning it identifies most heart disease cases correctly. However, accuracy is slightly lower compared to ensemble models and it may struggle with complex non-linear patterns. |
| Decision Tree | Achieves very high accuracy and precision by learning decision rules from data. Easy to interpret and visualize. However, it is prone to overfitting and may not generalize well to unseen data if tree depth is not controlled. |
| KNN | Provides balanced performance across accuracy, precision, and recall. Works well for datasets with clear class boundaries. Performance is highly dependent on the choice of K and feature scaling; computational cost increases with larger datasets. |
| Naive Bayes | Fast and efficient model with low computational cost. Performs reasonably well even with limited data. However, its assumption of feature independence reduces performance when features are correlated, which is common in medical datasets. |
| Random Forest | One of the most reliable models due to ensemble learning. Reduces overfitting by combining multiple decision trees. Handles non-linear relationships and feature interactions effectively. Provides stable and highly accurate predictions across different datasets. |
| XGBoost | Best performing model in this project with perfect scores across most metrics. Uses boosting to iteratively correct previous errors. Handles missing values and complex feature interactions well. Requires tuning but delivers superior predictive power and generalization. |

### 📊 Overall Insights

- Ensemble models (Random Forest, XGBoost) outperform single models.
- Logistic Regression is useful as a baseline and for interpretability.
- Decision Tree is highly accurate but must be regularized to avoid overfitting.
- KNN performance depends heavily on hyperparameter tuning.
- Naive Bayes is suitable for quick predictions and smaller datasets.
- XGBoost provides the most robust and production-ready performance for heart disease prediction.


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
- Predict heart disease risk for new patient data  


