# src/churnapp/predict.py
import joblib
import numpy as np
import pandas as pd
from .config import MODEL_PATH, FEATURES_PATH
from pathlib import Path

_model = None
_feature_names = None

def load_model():
    global _model, _feature_names
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _feature_names is None and Path(FEATURES_PATH).exists():
        _feature_names = joblib.load(FEATURES_PATH)
    return _model, _feature_names

def predict_single(input_df: pd.DataFrame):
    model, feature_names = load_model()
    # If model is a pipeline (preprocessor inside), pass raw input_df
    try:
        preds = model.predict(input_df)
        probs = model.predict_proba(input_df)[:,1] if hasattr(model, "predict_proba") else None
        return preds, probs
    except Exception:
        # fallback: one-hot encode input_df and align to feature_names
        if feature_names is None:
            raise RuntimeError("Model cannot accept raw df and feature names not available.")
        enc = pd.get_dummies(input_df)
        # align columns
        enc = enc.reindex(columns=feature_names, fill_value=0)
        preds = model.predict(enc)
        probs = model.predict_proba(enc)[:,1] if hasattr(model, "predict_proba") else None
        return preds, probs
