# 💳 Credit Card Fraud Detection (Machine Learning)

This project builds and evaluates machine learning models to detect fraudulent credit card transactions using the well-known **European credit card fraud dataset**.

It is an **end-to-end fraud detection pipeline** that covers:
- Data loading & exploration
- Class imbalance understanding
- Feature scaling
- Model training (Logistic Regression & Random Forest)
- Handling imbalance with **SMOTE**
- Evaluation using **precision, recall, F1-score, ROC–AUC**
- Saving the final model + scaler for deployment

---

## 🧾 Dataset

- Source: Public "Credit Card Fraud Detection" dataset  
- Rows: 284,807 transactions  
- Features:
  - `Time` – seconds from the first transaction
  - `V1`–`V28` – PCA components (anonymized features)
  - `Amount` – transaction amount
  - `Class` – target label  
    - `0` = legitimate transaction  
    - `1` = fraudulent transaction

> Note: Due to confidentiality, the original features were transformed using **PCA**, so `V1`–`V28` are anonymized components.

---

## 🧠 Problem

Fraud is **extremely rare** in the dataset:

- Legitimate (Class 0): 284,315+
- Fraud (Class 1): ~492

This makes the dataset **highly imbalanced**, so a naive model that always predicts “0” (non-fraud) would have 99.8% accuracy but be completely useless.

Because of this, we focus on:

- **Recall (for fraud)** – how many frauds we catch  
- **Precision (for fraud)** – how many predicted frauds are actually fraud  
- **F1-score & ROC–AUC** – balanced & global performance metrics  

---

## 📊 Exploratory Data Analysis (EDA)

Key steps:

1. **Class distribution**
   - Used `value_counts()` and `countplot` to visualize imbalance.
   - Fraud class is < 0.2% of total.

2. **Amount analysis**
   - Boxplots and histograms for `Amount` vs `Class`
   - Fraud transactions often cluster in **certain amount ranges**, typically smaller amounts.

3. **Time analysis**
   - Histograms of `Time` for fraud vs non-fraud
   - Fraud often occurs in **specific time windows**, e.g., low-activity periods.

4. **Correlation analysis**
   - Correlation heatmap using `df.corr()` and `seaborn.heatmap`
   - Some PCA components (e.g., `V10`, `V12`, `V14`, `V17`) show stronger correlation with fraud.

---

## 🏗 Feature Preparation

- Separated features and target:

  ```python
  X = df.drop("Class", axis=1)
  y = df["Class"]
