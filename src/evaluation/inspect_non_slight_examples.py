import os
import numpy as np
import pandas as pd

from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


CLEAN_DATA_PATH = "data/processed/accidents_india_clean.csv"
PREPROCESSOR_PATH = "models/ann_preprocessor.pkl"
MODEL_PATH = "models/ann_model.h5"
LABEL_ENCODER_PATH = "models/ann_label_encoder.pkl"


def main():
    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError(f"Clean data not found at {CLEAN_DATA_PATH}")

    df = pd.read_csv(CLEAN_DATA_PATH)

    if "Accident_Severity" not in df.columns:
        raise KeyError("Expected 'Accident_Severity' in cleaned data")

    y_text = df["Accident_Severity"]
    X = df.drop(columns=["Accident_Severity"])

    # Encode labels like in training
    le = LabelEncoder()
    y = le.fit_transform(y_text)

    # Same split as training
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = load(PREPROCESSOR_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    X_test_proc = preprocessor.transform(X_test)
    if hasattr(X_test_proc, "toarray"):
        X_test_proc = X_test_proc.toarray()

    probs = model.predict(X_test_proc, verbose=0)
    preds = np.argmax(probs, axis=1)
    pred_labels = le.inverse_transform(preds)

    X_test = X_test.reset_index(drop=True)
    y_test_labels = le.inverse_transform(y_test)

    # Columns you actually use in the form
    form_cols = [
        "Day_of_week",
        "Age_band_of_driver",
        "Sex_of_driver",
        "Weather_conditions",
        "Light_conditions",
        "Number_of_vehicles_involved",
        "Number_of_casualties",
        "Hour",
    ]

    for col in form_cols:
        if col not in X_test.columns:
            print(f"Warning: form column '{col}' not in X_test")
    
    print("\n=== Examples predicted as SERIOUS INJURY ===")
    serious_idx = np.where(pred_labels == "Serious Injury")[0][:5]
    for i in serious_idx:
        row = X_test.iloc[i]
        true_label = y_test_labels[i]
        p = probs[i]
        print("\n--- Example ---")
        print(f"True label: {true_label}")
        print(f"Predicted: Serious Injury")
        print(f"Probabilities: Fatal={p[0]:.3f}, Serious={p[1]:.3f}, Slight={p[2]:.3f}")
        print("Form JSON:")
        d = {col: row.get(col, None) for col in form_cols}
        print(d)

    print("\n=== Examples predicted as FATAL INJURY ===")
    fatal_idx = np.where(pred_labels == "Fatal injury")[0][:5]
    for i in fatal_idx:
        row = X_test.iloc[i]
        true_label = y_test_labels[i]
        p = probs[i]
        print("\n--- Example ---")
        print(f"True label: {true_label}")
        print(f"Predicted: Fatal injury")
        print(f"Probabilities: Fatal={p[0]:.3f}, Serious={p[1]:.3f}, Slight={p[2]:.3f}")
        print("Form JSON:")
        d = {col: row.get(col, None) for col in form_cols}
        print(d)


if __name__ == "__main__":
    main()
