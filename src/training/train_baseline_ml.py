import os
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from joblib import dump


CLEAN_DATA_PATH = "data/processed/accidents_india_clean.csv"
MODELS_DIR = "models"
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "baseline_preprocessor.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "baseline_model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "baseline_metadata.json")


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


def train_and_evaluate(
    name: str,
    model,
    preprocessor: ColumnTransformer,
    X_train,
    y_train,
    X_val,
    y_val,
):
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")

    print(f"\n=== {name} ===")
    print(f"Validation accuracy: {acc:.4f}")
    print(f"Validation macro F1: {f1:.4f}")
    print("Classification report:")
    print(classification_report(y_val, y_pred))

    return clf, acc, f1


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

    y = df["Accident_Severity"]
    X = df.drop(columns=["Accident_Severity"])

    # Train/val/test split: 60/20/20 (stratified)
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
        test_size=0.25,  # 0.25 of 0.8 = 0.2
        random_state=42,
        stratify=y_temp,
    )

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    preprocessor = build_preprocessor(X_train)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42,
        ),
    }

    best_name = None
    best_clf = None
    best_f1 = -1.0

    for name, model in models.items():
        clf, acc, f1 = train_and_evaluate(
            name,
            model,
            preprocessor,
            X_train,
            y_train,
            X_val,
            y_val,
        )
        if f1 > best_f1:
            best_f1 = f1
            best_clf = clf
            best_name = name

    if best_clf is None:
        raise RuntimeError("No model was successfully trained.")

    print(f"\nBest model on validation: {best_name} (F1={best_f1:.4f})")

    # Evaluate best model on test set
    y_test_pred = best_clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")

    print("\n=== Best model test performance ===")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test macro F1: {test_f1:.4f}")
    print("Test classification report:")
    print(classification_report(y_test, y_test_pred))

    # Extract the fitted preprocessor and model from the pipeline
    fitted_preprocessor = best_clf.named_steps["preprocessor"]
    fitted_model = best_clf.named_steps["model"]

    # Save preprocessor and model
    dump(fitted_preprocessor, PREPROCESSOR_PATH)
    dump(fitted_model, MODEL_PATH)

    metadata = {
        "model_type": best_name,
        "feature_columns": X.columns.tolist(),
        "target_name": "Accident_Severity",
        "classes": sorted(y.unique().tolist()),
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved preprocessor to: {PREPROCESSOR_PATH}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
