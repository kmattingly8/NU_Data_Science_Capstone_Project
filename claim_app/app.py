import os
from flask import Flask, render_template, request
import pandas as pd
import joblib
from preprocess import preprocess_pipeline  # import at the top

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model, scaler, expected columns
model = joblib.load(os.path.join(BASE_DIR, "model", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))
expected_columns = joblib.load(os.path.join(BASE_DIR, "model", "expected_columns.pkl"))

best_threshold = 0.34  

app = Flask(__name__)

def preprocess_user_input(form):
    # Convert form dict to DataFrame
    df = pd.DataFrame([form])

    # Convert date strings to datetime
    df['Appt_Date'] = pd.to_datetime(df['Appt_Date'])
    df['Initial_Claim_Submit_Date'] = pd.to_datetime(df['Initial_Claim_Submit_Date'])
    df['Final_Claim_Submit_Date'] = pd.to_datetime(df['Final_Claim_Submit_Date'])

    # Preprocess data using pipeline and scaler
    df_processed = preprocess_pipeline(df, scaler)

    # Add any missing expected columns with 0
    for col in expected_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0

    # Reorder columns to match model expectation
    df_processed = df_processed[expected_columns]

    return df_processed

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form = request.form.to_dict()
        input_data = preprocess_user_input(form)
        proba = model.predict_proba(input_data)[:, 1][0]
        prediction = int(proba >= best_threshold)

        return render_template("index.html", proba=round(proba, 4), prediction=prediction)

    return render_template("index.html", proba=None, prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
