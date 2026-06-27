
# LoanGuard AI - Loan Default Prediction

## Project Overview
Predicts loan default risk using Random Forest classifier on financial data.

## Dataset
- 10,000 records
- Features: Employed, Bank Balance, Annual Salary
- Target: Defaulted? (0/1)
- Class imbalance handled using class_weight='balanced'

## Model Performance
- Accuracy: 96%
- ROC-AUC: 0.87
- Algorithm: Random Forest (100 estimators)

## Key Findings
- Bank Balance is the strongest predictor of default
- Annual Salary is second most important feature
- Employment status has relatively lower predictive power

## Tech Stack
- Python, Pandas, Scikit-learn, Matplotlib, Seaborn
- Google Colab for training
