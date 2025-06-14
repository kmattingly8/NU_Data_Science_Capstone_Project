import pandas as pd
import numpy as np
import joblib
import os

def get_fiscal_quarter(date):
    if pd.isnull(date):
        return None
    month = date.month
    if 1 <= month <= 3:
        return 'Q1'
    elif 4 <= month <= 6:
        return 'Q2'
    elif 7 <= month <= 9:
        return 'Q3'
    else:
        return 'Q4'

def preprocess_pipeline(df, scaler):
    # Load saved artifacts
    scaler = joblib.load(os.path.join('model', 'scaler.pkl'))
    expected_cols = joblib.load(os.path.join('model', 'expected_columns.pkl'))

    # --- Date Features ---
    df['Appt_Date'] = pd.to_datetime(df['Appt_Date'])
    df['Initial_Claim_Submit_Date'] = pd.to_datetime(df['Initial_Claim_Submit_Date'])
    df['Final_Claim_Submit_Date'] = pd.to_datetime(df['Final_Claim_Submit_Date'])

    df['Year_Service'] = df['Appt_Date'].dt.year
    df['Month_Service'] = df['Appt_Date'].dt.month
    df['Day_of_Week_Service'] = df['Appt_Date'].dt.weekday
    df['Initial_Submission_Delay'] = (df['Initial_Claim_Submit_Date'] - df['Appt_Date']).dt.days
    df['Final_Submission_Delay'] = (df['Final_Claim_Submit_Date'] - df['Appt_Date']).dt.days
    df['Quarter_Service'] = df['Appt_Date'].apply(get_fiscal_quarter)

    # --- COVID Flags ---
    df['pre_COVID'] = (df['Year_Service'] < 2020).astype(int)
    df['COVID_era'] = ((df['Year_Service'] >= 2020) & (df['Year_Service'] <= 2022)).astype(int)
    df['post_COVID'] = (df['Year_Service'] > 2022).astype(int)

    # --- Appointment Type Mapping ---
    appttype_mapping = {
        'ANY 20': 'Follow Up', 'Established Patient': 'Follow Up',
        'Medicare Annual Wellness': 'Annual Visit',
        'Procedure (cash)': 'Other', 'Hormone Replacement': 'Other',
        'Lab Work': 'Other', 'Biofeedback Therapy': 'Other',
        'functional medicine consult': 'Other', 'COGNITIVE CONSULTATION': 'Other',
        'VACCINATION': 'Other', 'hair removal': 'Other',
        'cosmetic treatments': 'Other', 'Preliminary Consultation': 'Other',
        'Membership visit': 'Other'
    }
    df['Appt_Type'] = df['Appt_Type'].replace(appttype_mapping)

    # --- ICD Code Count ---
    icd_cols = [f'ICD10DiagCode_{i}' for i in range(1, 13)]
    df['ICD10_Code_Count'] = df[icd_cols].notna().sum(axis=1)
    df[icd_cols] = df[icd_cols].fillna("None")

    # --- Procedure + ICD Combos ---
    for i in range(1, 13):
        icd_col = f'ICD10DiagCode_{i}'
        combo_col = f'Procedure_ICD10_{i}_Combo'
        df[combo_col] = np.where(
            df[icd_col] != "None",
            df['Procedure_Code'].astype(str) + '_' + df[icd_col].astype(str),
            "None"
        )

    # --- Procedure + Charge Pair ---
    df['Total_Charge'] = pd.to_numeric(df['Total_Charge'], errors='coerce')
    df['Procedure_Charge_Pair'] = (
    df['Procedure_Code'].astype(str) + '_' + 
    df['Total_Charge'].round(2).astype(str)
)


    # --- Missing Value Handling ---
    df['Appt_Type'] = df['Appt_Type'].fillna('None')
    df['RAF_Score'] = pd.to_numeric(df['RAF_Score'], errors='coerce')
    df['RAF_Score'] = df['RAF_Score'].fillna(df['RAF_Score'].median())
    df['Initial_Submission_Delay'] = df['Initial_Submission_Delay'].fillna(df['Initial_Submission_Delay'].median())
    df['Final_Submission_Delay'] = df['Final_Submission_Delay'].fillna(df['Final_Submission_Delay'].median())

    # --- Binary Mapping ---
    df['Patient_Sex'] = df['Patient_Sex'].map({'M': 0, 'F': 1})

    # --- Cyclical Encoding ---
    df['Month_Service_sin'] = np.sin(2 * np.pi * df['Month_Service'] / 12)
    df['Month_Service_cos'] = np.cos(2 * np.pi * df['Month_Service'] / 12)
    df['Weekday_Service_sin'] = np.sin(2 * np.pi * df['Day_of_Week_Service'] / 7)
    df['Weekday_Service_cos'] = np.cos(2 * np.pi * df['Day_of_Week_Service'] / 7)

    # --- One-Hot Encoding ---
    categorical_cols = ['Primary_Insurance_Type', 'Secondary_Insurance_Type', 'Appt_Type', 'Quarter_Service']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # --- Align Columns to Training ---
    df = df.reindex(columns=expected_cols, fill_value=0)

    # --- Scale Numerical Features ---
    numeric_cols = ['Total_Charge', 'Patient_Age', 'RAF_Score',
                    'Initial_Submission_Delay', 'Final_Submission_Delay', 'ICD10_Code_Count']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # --- Drop Original Date Columns ---
    df.drop(['Appt_Date', 'Initial_Claim_Submit_Date', 'Final_Claim_Submit_Date',
             'Year_Service', 'Month_Service', 'Day_of_Week_Service'], axis=1, inplace=True, errors='ignore')

    return df


