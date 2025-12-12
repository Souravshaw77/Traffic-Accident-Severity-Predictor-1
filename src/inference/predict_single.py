import os
import json
import numpy as np
import pandas as pd
from joblib import load
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "ann_preprocessor.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "ann_model.h5")
METADATA_PATH = os.path.join(MODELS_DIR, "ann_metadata.json")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "ann_label_encoder.pkl")


# Load artifacts once
with open(METADATA_PATH, "r") as f:
    METADATA = json.load(f)

FEATURE_COLUMNS = METADATA["feature_columns"]
CLASS_LABELS = METADATA["classes"]

PREPROCESSOR = load(PREPROCESSOR_PATH)
MODEL = tf.keras.models.load_model(MODEL_PATH)


def _build_input_dataframe(input_data: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame in the same column order as training.
    Missing values are set to None (handled by preprocessor imputers).
    """
    row = {}
    for col in FEATURE_COLUMNS:
        row[col] = input_data.get(col, None)
    return pd.DataFrame([row])


def _compute_risk(probabilities: dict) -> tuple[float, str]:
    """
    Compute risk_score = P(Fatal) + P(Serious)
    Map to Low / Medium / High based on thresholds.
    """
    p_fatal = float(probabilities.get("Fatal injury", 0.0))
    p_serious = float(probabilities.get("Serious Injury", 0.0))

    risk_score = p_fatal + p_serious  # between 0 and 1

    if risk_score >= 0.7:
        risk_level = "High"
    elif risk_score >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return risk_score, risk_level


def predict_severity(input_data: dict) -> dict:
    """
    Main inference function used by Flask and CLI.
    Returns:
      - predicted_label
      - probabilities per class
      - risk_score (0–1)
      - risk_level (Low / Medium / High)
    """
    df_input = _build_input_dataframe(input_data)

    X_proc = PREPROCESSOR.transform(df_input)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    probs = MODEL.predict(X_proc, verbose=0)[0]

    # Map probabilities to class labels
    prob_dict = {
        CLASS_LABELS[i]: float(probs[i]) for i in range(len(CLASS_LABELS))
    }

    # Argmax label
    predicted_index = int(np.argmax(probs))
    predicted_label = str(CLASS_LABELS[predicted_index])

    # Risk computation
    risk_score, risk_level = _compute_risk(prob_dict)

    return {
        "predicted_label": predicted_label,
        "probabilities": prob_dict,
        "risk_score": risk_score,
        "risk_level": risk_level,
    }


if __name__ == "__main__":
    # Quick manual test
    sample = {
        "Day_of_week": "Friday",
        "Age_band_of_driver": "18-30",
        "Sex_of_driver": "Male",
        "Weather_conditions": "Raining",
        "Light_conditions": "Darkness - lights lit",
        "Number_of_vehicles_involved": 3,
        "Number_of_casualties": 3,
        "Hour": 22,
    }
    res = predict_severity(sample)
    print(json.dumps(res, indent=2))
