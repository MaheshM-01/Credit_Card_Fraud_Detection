# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using advanced classification techniques and imbalanced data handling.

##  Project Overview

This project implements a comprehensive fraud detection system that identifies fraudulent credit card transactions from a highly imbalanced dataset. The solution employs XGBoost with SMOTE oversampling and achieves strong performance metrics including 91% precision and 81% recall on the fraud class.

##  Key Features

- **Comprehensive EDA**: In-depth exploratory data analysis with class distribution, temporal patterns, and feature correlation analysis
- **Advanced Preprocessing**: PCA-based feature processing with robust scaling for optimal model performance
- **Imbalanced Data Handling**: SMOTE (Synthetic Minority Over-sampling Technique) to address the 99.83% to 0.17% class imbalance
- **Hyperparameter Optimization**: Optuna-based automated tuning with 30 trials to maximize PR-AUC
- **Custom Threshold Selection**: F2-score optimization for better recall in fraud detection
- **Model Persistence**: Trained pipeline saved for deployment and inference

##  Dataset

The dataset contains credit card transactions with:
- **284,807 transactions** (after removing 1,081 duplicates)
- **30 features**: 28 PCA-transformed features (V1-V28), Time, and Amount
- **Highly imbalanced**: 0.17% fraudulent transactions
- **No missing values**

##  Technologies Used

- **Python 3.x**
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn, missingno
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Imbalanced Learning**: imbalanced-learn (SMOTE)
- **Hyperparameter Tuning**: Optuna
- **Model Persistence**: joblib


## 🔍 Exploratory Data Analysis

Key insights from EDA:

- **Class Distribution**: Severe imbalance with 99.83% legitimate vs 0.17% fraudulent transactions
- **Temporal Patterns**: Transaction patterns analyzed across 48 hours with hourly fraud distribution
- **Amount Analysis**: Fraud transactions show different amount distributions compared to legitimate ones
- **Feature Correlations**: Identified top correlated features (V17, V14, V12, V10) with fraud class

##  Model Development

### Preprocessing Pipeline

1. **PCA Features (V1-V28)**:
   - Median imputation
   - PCA with 95% variance retention

2. **Raw Features (Amount, Time)**:
   - Median imputation
   - Robust scaling (resistant to outliers)

### Model Selection & Training

- **Algorithm**: XGBoost Classifier
- **Resampling**: SMOTE for handling class imbalance
- **Cross-Validation**: 5-fold Stratified K-Fold
- **Optimization Metric**: PR-AUC (Precision-Recall Area Under Curve)

### Best Hyperparameters

```python
{
    'n_estimators': 418,
    'max_depth': 8,
    'learning_rate': 0.1547,
    'subsample': 0.8587,
    'colsample_bytree': 0.8160,
    'scale_pos_weight': 10
}
```

## 📈 Model Performance

### Test Set Results (Threshold: 0.9792)

 Metric  Score 

 **ROC-AUC**  0.9750 
 **PR-AUC**  0.8383 
 **Precision (Fraud)**  0.91 
 **Recall (Fraud)**  0.81 
 **F1-Score (Fraud)**  0.85 
 **F2-Score**  0.8308 

### Confusion Matrix

 Predicted Legitimate  Predicted Fraud 

| **Actual Legitimate**  56,856  8 
| **Actual Fraud**  19  79 

- **False Positives**: 8 (minimal inconvenience to customers)
- **False Negatives**: 19 (missed frauds, ~19% of fraud cases)




##  Key Insights

1. **Threshold Tuning Matters**: Using F2-score optimization improved recall while maintaining high precision
2. **SMOTE Effectiveness**: Oversampling the minority class significantly improved model learning
3. **Feature Engineering**: PCA features were already provided, but combining with robust scaling for raw features enhanced performance
4. **Class Imbalance**: Addressing the 600:1 imbalance ratio was critical for success

##  Future Improvements

- [ ] Implement ensemble methods (stacking multiple models)
- [ ] Explore deep learning approaches (LSTM for temporal patterns)
- [ ] Add real-time prediction API
- [ ] Implement model monitoring and drift detection
- [ ] Cost-sensitive learning based on fraud amounts
- [ ] Feature importance analysis and explainability (SHAP values)


##  Contact
**Email**: mahesh0105.m@gmail.com
**linkedin**: https://www.linkedin.com/in/mahesh-m0105

