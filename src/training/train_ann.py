import os
import json

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


CLEAN_DATA_PATH = "data/processed/accidents_india_clean.csv"
MODELS_DIR = "models"
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "ann_preprocessor.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "ann_model.h5")
METADATA_PATH = os.path.join(MODELS_DIR, "ann_metadata.json")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "ann_label_encoder.pkl")

# 🔹 EXACT features used in the web form
USED_FEATURES = [
    "Day_of_week",
    "Age_band_of_driver",
    "Sex_of_driver",
    "Weather_conditions",
    "Light_conditions",
    "Number_of_vehicles_involved",
    "Number_of_casualties",
    "Hour",
]


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def build_ann(input_dim: int, num_classes: int) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    ensure_dirs()

    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError(
            f"Clean data not found at {CLEAN_DATA_PATH}. "
            f"Run data_preprocessing/data_prep.py first."
        )

    df = load_data(CLEAN_DATA_PATH)
    if "Accident_Severity" not in df.columns:
        raise KeyError("Expected 'Accident_Severity' column in cleaned data.")

    # ✅ Only keep the features that UI actually sends
    missing = [c for c in USED_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"These USED_FEATURES are missing in data: {missing}")

    y_text = df["Accident_Severity"]
    X = df[USED_FEATURES].copy()

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=42,
        stratify=y_temp,
    )

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    preprocessor = build_preprocessor(X_train)

    # Fit preprocessor only on training data
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_val_proc = X_val_proc.toarray()
        X_test_proc = X_test_proc.toarray()

    input_dim = X_train_proc.shape[1]
    num_classes = len(label_encoder.classes_)

    print(f"Input dim after preprocessing: {input_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")

    # SMOTE on training data
    print("\nClass distribution BEFORE SMOTE (y_train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {cls} ({label_encoder.inverse_transform([cls])[0]}): {cnt}")

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_proc, y_train)

    print("\nClass distribution AFTER SMOTE (y_train_res):")
    unique_res, counts_res = np.unique(y_train_res, return_counts=True)
    for cls, cnt in zip(unique_res, counts_res):
        print(f"  {cls} ({label_encoder.inverse_transform([cls])[0]}): {cnt}")

    model = build_ann(input_dim, num_classes)

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    ckpt = callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    model.fit(
        X_train_res,
        y_train_res,
        validation_data=(X_val_proc, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[es, ckpt],
        verbose=1,
    )

    # Validation
    val_probs = model.predict(X_val_proc, verbose=0)
    val_pred = np.argmax(val_probs, axis=1)

    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average="macro")

    print("\n=== ANN (SMOTE, 8 features) Validation Performance ===")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation macro F1: {val_f1:.4f}")
    print("Validation classification report:")
    print(classification_report(y_val, val_pred, target_names=label_encoder.classes_))

    # Test
    test_probs = model.predict(X_test_proc, verbose=0)
    test_pred = np.argmax(test_probs, axis=1)

    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="macro")

    print("\n=== ANN (SMOTE, 8 features) Test Performance ===")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test macro F1: {test_f1:.4f}")
    print("Test classification report:")
    print(classification_report(y_test, test_pred, target_names=label_encoder.classes_))

    # Save artifacts
    dump(preprocessor, PREPROCESSOR_PATH)

    metadata = {
        "model_type": "ANN_SMOTE_8_features",
        "input_dim": int(input_dim),
        "feature_columns": USED_FEATURES,  # 🔹 only these 8 now
        "target_name": "Accident_Severity",
        "classes": label_encoder.classes_.tolist(),
        "label_encoder_classes": label_encoder.classes_.tolist(),
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    dump(label_encoder, LABEL_ENCODER_PATH)

    print(f"\nSaved ANN preprocessor to: {PREPROCESSOR_PATH}")
    print(f"Saved ANN model to: {MODEL_PATH}")
    print(f"Saved ANN metadata to: {METADATA_PATH}")
    print(f"Saved ANN label encoder to: {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()
