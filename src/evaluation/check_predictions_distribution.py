import pandas as pd
import numpy as np
from joblib import load
import tensorflow as tf

CLEAN_DATA_PATH = "data/processed/accidents_india_clean.csv"
PREPROCESSOR_PATH = "models/ann_preprocessor.pkl"
MODEL_PATH = "models/ann_model.h5"
LABEL_ENCODER_PATH = "models/ann_label_encoder.pkl"

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main():
    df = pd.read_csv(CLEAN_DATA_PATH)
    y_text = df["Accident_Severity"]
    X = df.drop(columns=["Accident_Severity"])

    le = LabelEncoder()
    y = le.fit_transform(y_text)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = load(PREPROCESSOR_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    X_test_proc = preprocessor.transform(X_test)
    if hasattr(X_test_proc, "toarray"):
        X_test_proc = X_test_proc.toarray()

    probs = model.predict(X_test_proc, verbose=0)
    preds = np.argmax(probs, axis=1)

    _, counts = np.unique(preds, return_counts=True)
    print("Predicted class distribution on test set:")
    for cls_idx, cnt in zip(np.unique(preds), counts):
        print(f"{cls_idx} ({le.inverse_transform([cls_idx])[0]}): {cnt}")


if __name__ == "__main__":
    main()
