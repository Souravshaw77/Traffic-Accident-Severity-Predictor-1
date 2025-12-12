import os
import pandas as pd


RAW_DATA_PATH = "data/raw/road_accidents_india.csv"  # put your Kaggle CSV here
PROCESSED_DIR = "data/processed"
CLEAN_DATA_PATH = os.path.join(PROCESSED_DIR, "accidents_india_clean.csv")


def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def detect_column(df: pd.DataFrame, candidates):
    """Return the first existing column from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Try to detect target column
    target_candidates = [
        "Accident_Severity",
        "Accident_severity",
        "Accident severity",
        "Severity",
        "Accident_Severity_Code"
    ]
    target_col = detect_column(df, target_candidates)
    if target_col is None:
        raise KeyError(
            f"Could not find target column. "
            f"Tried: {target_candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    # Try to detect time column
    time_candidates = ["Time", "Time_of_Accident", "Accident_Time", "Accident Time"]
    time_col = detect_column(df, time_candidates)

    # Drop rows without target
    df = df.dropna(subset=[target_col])

    # Map severity labels into Low/Medium/High if possible
    severity_map = {
        "Slight": "Low",
        "SLIGHT": "Low",
        "slight": "Low",
        "Serious": "Medium",
        "SERIOUS": "Medium",
        "serious": "Medium",
        "Fatal": "High",
        "FATAL": "High",
        "fatal": "High",
    }

    def map_severity(x):
        s = str(x).strip()
        return severity_map.get(s, s)

    df[target_col] = df[target_col].apply(map_severity)

    # Rename target column to a standard name
    df = df.rename(columns={target_col: "Accident_Severity"})

    # Extract hour from time column if present
    if time_col is not None:
        def extract_hour(x):
            try:
                # Expect "HH:MM" or "H:MM"
                return int(str(x).split(":")[0])
            except Exception:
                return None

        df["Hour"] = df[time_col].apply(extract_hour)
        df = df.dropna(subset=["Hour"])

    # Drop fully empty columns
    df = df.dropna(axis=1, how="all")

    # Reset index
    df = df.reset_index(drop=True)

    return df


def main():
    ensure_dirs()

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Raw data not found at {RAW_DATA_PATH}. "
            f"Place your Road Accident Severity in India CSV there."
        )

    print(f"Loading raw data from: {RAW_DATA_PATH}")
    df_raw = load_raw_data(RAW_DATA_PATH)
    print(f"Raw shape: {df_raw.shape}")
    print(f"Columns: {list(df_raw.columns)}")

    df_clean = clean_data(df_raw)
    print(f"Cleaned shape: {df_clean.shape}")

    df_clean.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Saved cleaned data to: {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()
