# train_model.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "dataset" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

def load_and_clean(path):
    df = pd.read_csv(path)
    # Convert TotalCharges to numeric (Telco quirk)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    # Map Churn to 0/1
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes":1, "No":0})
    return df

def build_pipeline(df):
    X = df.drop(["customerID", "Churn"], axis=1, errors="ignore")
    y = df["Churn"].astype(int)

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    return pipeline, X, y

def train():
    df = load_and_clean(DATA_PATH)
    pipeline, X, y = build_pipeline(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:,1]
    print("Classification report:\n", classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))

    # Save pipeline (preprocessor + model). Also save columns info for manual encoding if needed.
    joblib.dump(pipeline, MODEL_DIR / "churn_model.pkl")
    # Save encoded columns (fit transform X to get feature names)
    # Get feature names after preprocessing
    pre = pipeline.named_steps["preprocessor"]
    # numeric names same as numeric_cols
    num_cols = pre.transformers_[0][2]
    # get onehot names
    cat_transformer = pre.transformers_[1][1]
    cat_cols = pre.transformers_[1][2]
    cat_names = cat_transformer.get_feature_names_out(cat_cols).tolist()
    feature_names = list(num_cols) + cat_names
    joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")
    print("Saved model and feature names to model/")

if __name__ == "__main__":
    train()
