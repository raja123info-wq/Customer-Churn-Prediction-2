# src/churnapp/app.py
from flask import Flask, render_template, request, redirect, url_for, flash
from .config import DATASET_PATH
from .preprocessing import clean_total_charges, prepare_input_dataframe
from .predict import predict_single
import pandas as pd
from pathlib import Path

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = "change-me-to-secure"  # replace with env var in production

    # load sample dataset to get feature template
    if Path(DATASET_PATH).exists():
        df = pd.read_csv(DATASET_PATH)
        df = clean_total_charges(df)
    else:
        df = pd.DataFrame()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        # build form fields from dataset if available
        if request.method == "POST":
            # collect form data
            input_data = {}
            for k, v in request.form.items():
                # convert numeric-like fields
                if v.isnumeric():
                    input_data[k] = float(v)
                else:
                    # allow empty -> None
                    input_data[k] = v
            # prepare dataframe matching training features (drop churn)
            if not df.empty:
                template_df = df.drop(columns=["Churn"]) if "Churn" in df.columns else df
                input_df = prepare_input_dataframe(input_data, template_df)
            else:
                input_df = pd.DataFrame([input_data])
            try:
                preds, probs = predict_single(input_df)
                pred = preds[0]
                prob = probs[0] if probs is not None else None
                # map numeric pred to label
                label = "Yes" if (isinstance(pred, (int, float)) and pred == 1) or (isinstance(pred, str) and pred.lower().startswith("y")) else "No"
                return render_template("result.html", prediction=label, probability=prob)
            except Exception as e:
                flash(f"Prediction failed: {e}", "danger")
                return redirect(url_for("predict"))
        # GET: render form. Provide a few example fields (common telco columns)
        # Show fields only if dataset available to extract unique values
        fields = {}
        if not df.empty:
            # pick commonly used columns
            common = ["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","InternetService","Contract","PaymentMethod","MonthlyCharges","TotalCharges"]
            for c in common:
                if c in df.columns:
                    fields[c] = sorted(df[c].dropna().unique().tolist())
        return render_template("predict.html", fields=fields)

    return app
