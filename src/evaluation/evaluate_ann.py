import os
import json

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# FIXED: go 3 levels up to reach real project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLEAN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "accidents_india_clean.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "ann_preprocessor.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "ann_model.h5")
METADATA_PATH = os.path.join(MODELS_DIR, "ann_metadata.json")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "ann_label_encoder.pkl")



def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def load_artifacts():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    preprocessor = load(PREPROCESSOR_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    label_encoder: LabelEncoder = load(LABEL_ENCODER_PATH)

    feature_columns = metadata["feature_columns"]
    classes = metadata["classes"]

    return preprocessor, model, label_encoder, feature_columns, classes


def load_data(feature_columns, target_name="Accident_Severity"):
    df = pd.read_csv(CLEAN_DATA_PATH)
    if target_name not in df.columns:
        raise KeyError(f"Expected target column '{target_name}' in cleaned data.")

    y_text = df[target_name]
    X = df[feature_columns].copy()

    return X, y_text


def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("ANN (SMOTE, 8 features) – Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ensure_dirs()

    preprocessor, model, label_encoder, feature_columns, classes = load_artifacts()
    X, y_text = load_data(feature_columns, target_name="Accident_Severity")

    # Encode labels
    y = label_encoder.transform(y_text)

    # Reproduce same split as in train_ann.py
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # We only care about test set here
    X_test_proc = preprocessor.transform(X_test)
    if hasattr(X_test_proc, "toarray"):
        X_test_proc = X_test_proc.toarray()

    # Predict
    probs = model.predict(X_test_proc, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n=== ANN (SMOTE, 8 features) – Test Set Evaluation ===")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    cm_path = os.path.join(REPORTS_DIR, "ann_confusion_matrix_test.png")
    plot_confusion_matrix(cm, classes, cm_path)
    print(f"\nSaved confusion matrix image to: {cm_path}")


if __name__ == "__main__":
    main()
