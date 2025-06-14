# Improving Medical Claim Outcomes in a Small Clinical Setting:  

An Interpretable Machine Learning Approach to Predicting Insurance Denials

## Overview

This repository contains the code, data pipeline, and model development artifacts for a capstone project focused on predicting medical insurance claim denials using interpretable machine learning (ML) techniques. The project centers on a small, resource-constrained outpatient clinic setting and evaluates multiple supervised ML models for deployment feasibility, interpretability, and predictive accuracy.

## Objectives

- To determine whether ML models can accurately predict claim denials using structured claims data from a small outpatient clinic.
- To compare the predictive performance of interpretable models (e.g., logistic regression, EBM) with more complex approaches (e.g., ANN).
- To identify key predictors of claim denial outcomes (e.g., coding, insurance type, timing).
- To demonstrate the feasibility of deploying an interpretable ML model via a lightweight web app suitable for small clinics.

## Data Description

- **Source**: De-identified structured EHR and billing data from a solo physician clinic in Southern California.
- **Size**: 11,093 claims over ~10 years (2015–2024).
- **Features**: 25+ variables including patient demographics, procedure and diagnosis codes, payer types, visit timing, RAF scores, and submission delays.
- **Target**: Binary classification (`Denied = 1`, `Accepted = 0`).

> **Note**: Due to HIPAA regulations, raw data is not publicly shared. Only simulated or schema-representative samples may be included here for reference.

## Methodology

The ML workflow follows the full ML lifecycle, including:

### 1. **Data Preprocessing**
- Null handling and imputations (median for numerics; placeholders for categoricals).
- Feature engineering (e.g., Procedure–ICD pairing, delay features).
- One-hot, target, and cyclical encoding.
- Standardization of numeric features.
- Train-test split (70/30) and stratified k-fold cross-validation (5-fold).

### 2. **Model Development**
Models used:
- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- Explainable Boosting Machine (EBM)
- Artificial Neural Network (ANN)

Frameworks:
- `scikit-learn`
- `InterpretML`
- `Keras` / `TensorFlow`
- `SHAP` for model interpretability

### 3. **Handling Class Imbalance**
- SMOTE (Synthetic Minority Oversampling)
- Class weighting

### 4. **Model Evaluation**
- Primary metric: **ROC-AUC**
- Additional: Precision, Recall, F1-score, Accuracy
- Interpretability tools: SHAP values, feature importances, PDPs, global explanation graphs


## 📊 Model Performance Summary

Six machine learning models were trained and evaluated using ~11,000 claims. Performance was measured using **AUC**, **F1 score**, **accuracy**, **recall**, and **precision**.

| Model                  | AUC   | F1 Score | Recall | Precision | Accuracy |
|------------------------|-------|----------|--------|-----------|----------|
| ANN                 | 0.760 | 0.445    | 0.451  | 0.439     | 0.862    |
| Logistic Regression | 0.759 | 0.442    | **0.577**  | 0.359     | 0.821    |
| EBM                 | 0.698 | 0.341    | 0.446  | 0.276     | 0.789    |
| Random Forest       | 0.718 | 0.342    | 0.507  | 0.259     | 0.759    |
| AdaBoost            | 0.684 | 0.319    | 0.407  | 0.263     | 0.786    |
| Decision Tree       | 0.585 | 0.264    | 0.351  | 0.211     | 0.758    |

### Highlights:

- **ANN** had the best raw metrics but lacked interpretability.
- **Logistic Regression** performed nearly identically and had the **highest recall**, critical for flagging denied claims.
- **EBM** provided strong performance while maintaining **full interpretability** for clinic staff.

---

## Top Predictive Features

Across models, the following features were most predictive of claim denials:

- **CPT–ICD10 code compatibility** (`Procedure_ICD10_X_Combo`)
- **Final submission delay** (`Final_Submission_Delay`)
- **ICD10 diagnosis code count**
- **Appointment type** (e.g., new vs. established patient)
- **Payer category and insurance type**
- **COVID-era timing flags** (e.g., `post_COVID`)

> These features align with known drivers of denials such as coding mismatches and submission delays.

---

## 🔍 Model Interpretability

This project emphasized models that balance **accuracy** and **interpretability**.

- **Logistic Regression** and 🔍 **EBM** matched ANN in performance but provided **transparent decision-making**.
- **EBM** allowed visualization of individual feature contributions (e.g., how delays affect denial risk).
- **ANN** required SHAP values for post hoc interpretation, which are **less intuitive** for clinical staff.

> **Conclusion**: Interpretable models are both effective and practical in low-resource healthcare settings.

---

## Deployment

A lightweight, local Flask web application was developed for real-time denial risk scoring. **Features include:**

- **Runs locally** to preserve patient privacy (**HIPAA-compliant**) - no external data transmission
- **Simple UI** for non-technical users (clinic staff) - input claim data through a web form
- **Accepts structured input** and returns denial risk prediction

>  Prototype only. Not currently in production.


## Ethical & Compliance

- All data were de-identified and analyzed under HIPAA’s Limited Data Set (LDS) guidelines.
- IRB exemption granted by National University.
- All team members completed human subjects research training (CITI Program).
- A signed Data Use Agreement (DUA) was executed.

## Repository Structure

├── data/ # Sample schema or synthetic data
├── notebooks/ # Jupyter Notebooks for EDA and modeling
├── models/ # Saved model files (.joblib or .h5)
├── app/ # Flask app for deployment
├── scripts/ # Python scripts for preprocessing, training
├── requirements.txt # Python dependencies
└── README.md # This file


