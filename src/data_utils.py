"""
data_utils.py
-------------
Handles data loading, cleaning, encoding, and train/test splitting
for the Telecom Churn prediction pipeline.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

RANDOM_STATE = 42

BINARY_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "Churn",
]

CATEGORICAL_COLUMNS = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
]

NUMERIC_COLUMNS = [
    "tenure", "MonthlyCharges", "TotalCharges",
]

TARGET_COLUMN = "Churn"


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw Telco churn CSV into a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all original columns preserved.

    Raises
    ------
    FileNotFoundError
        If the specified filepath does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    print(f"[data_utils] Loaded {len(df):,} rows and {df.shape[1]} columns from '{filepath}'.")
    return df


def load_from_upload(uploaded_file) -> "pd.DataFrame":
    """
    Load a CSV from a Streamlit UploadedFile object (BytesIO-compatible).

    This is the entry point for user-uploaded data in the dashboard.
    Validates the file is non-empty and parseable before returning.

    Parameters
    ----------
    uploaded_file : streamlit UploadedFile
        The file object returned by st.file_uploader().

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If the uploaded file cannot be parsed or is empty.
    """
    import pandas as _pd
    try:
        df = _pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"Could not read uploaded CSV: {e}")

    if df.empty:
        raise ValueError("Uploaded CSV is empty.")

    print(f"[data_utils] Uploaded file: {len(df):,} rows, {df.shape[1]} columns.")
    return df


def validate_uploaded_columns(df, required_cols=None):
    """
    Verify minimum required columns exist in an uploaded DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    required_cols : list, optional
        Defaults to ["tenure", "MonthlyCharges"].

    Returns
    -------
    list
        Names of missing columns. Empty list means validation passed.
    """
    if required_cols is None:
        required_cols = ["tenure", "MonthlyCharges"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[data_utils] Missing columns in upload: {missing}")
    else:
        print("[data_utils] Upload column validation passed.")
    return missing



# ──────────────────────────────────────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce TotalCharges to numeric, filling non-parseable values with the
    product of tenure and MonthlyCharges (a reasonable imputation for new
    customers who have a blank TotalCharges entry).

    Parameters
    ----------
    df : pd.DataFrame
        Raw or partially processed DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with TotalCharges as float64.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    missing_mask = df["TotalCharges"].isna()
    n_missing = missing_mask.sum()
    if n_missing > 0:
        df.loc[missing_mask, "TotalCharges"] = (
            df.loc[missing_mask, "tenure"] * df.loc[missing_mask, "MonthlyCharges"]
        )
        print(f"[data_utils] Imputed {n_missing} missing TotalCharges values.")

    return df


def remove_duplicates(df: pd.DataFrame, id_col: str = "customerID") -> pd.DataFrame:
    """
    Drop duplicate rows based on the customer ID column.

    Parameters
    ----------
    df : pd.DataFrame
    id_col : str
        Column to use for deduplication.

    Returns
    -------
    pd.DataFrame
    """
    before = len(df)
    df = df.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[data_utils] Removed {dropped} duplicate rows.")
    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Run a basic data quality report: missing values, dtypes, and value ranges.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        Dictionary containing missing counts, dtypes summary, and numeric stats.
    """
    report = {
        "shape": df.shape,
        "missing_counts": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_summary": df.describe().to_dict(),
    }
    total_missing = sum(v for v in report["missing_counts"].values())
    print(f"[data_utils] Data quality check: {df.shape[0]:,} rows, "
          f"{df.shape[1]} cols, {total_missing} total missing values.")
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Encoding
# ──────────────────────────────────────────────────────────────────────────────

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all binary and multi-class categorical columns.

    Binary columns are mapped to 0/1.
    Multi-class categorical columns receive one-hot encoding with
    drop_first=True to avoid multicollinearity.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with encoded features and no original categorical strings.
    """
    df = df.copy()

    # Binary columns
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(binary_map).astype(int)

    # Senior citizen is already 0/1 in the base dataset
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # One-hot encode multi-class categoricals
    cols_present = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    df = pd.get_dummies(df, columns=cols_present, drop_first=True)

    # Convert all boolean columns from get_dummies to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Drop display-only engineered columns that are non-numeric and not needed for modeling.
    # complaint_text_clean is intentionally KEPT here so build_tfidf_features() can
    # consume it after split_train_test(). It will be dropped inside build_tfidf_features.
    NON_MODEL_COLS = ["tenure_bracket", "complaint_text"]
    df = df.drop(columns=[c for c in NON_MODEL_COLS if c in df.columns], errors="ignore")

    # Final safety net: drop any remaining non-numeric columns EXCEPT complaint_text_clean,
    # which must pass through to build_tfidf_features().
    PRESERVE_COLS = {"complaint_text_clean"}
    remaining_obj = [
        c for c in df.select_dtypes(include=["object", "string", "category"]).columns
        if c not in PRESERVE_COLS
    ]
    if remaining_obj:
        print(f"[data_utils] Dropping remaining non-numeric columns: {remaining_obj}")
        df = df.drop(columns=remaining_obj)

    print(f"[data_utils] Encoding complete. DataFrame now has {df.shape[1]} columns.")
    return df


def scale_numeric_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: list = None,
) -> tuple:
    """
    Fit a StandardScaler on X_train and transform both X_train and X_test.
    Only applies to numeric columns relevant for linear models.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    numeric_cols : list, optional
        Column names to scale. Defaults to NUMERIC_COLUMNS plus engineered
        numeric features.

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    if numeric_cols is None:
        numeric_cols = [
            c for c in NUMERIC_COLUMNS + [
                "avg_monthly_charges",
                "tenure_network_interaction",
                "charges_network_interaction",
                "service_bundle_count",
            ]
            if c in X_train.columns
        ]

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print(f"[data_utils] Scaled {len(numeric_cols)} numeric columns.")
    return X_train, X_test, scaler


# ──────────────────────────────────────────────────────────────────────────────
# Train / Test Split
# ──────────────────────────────────────────────────────────────────────────────

def split_train_test(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    test_size: float = 0.20,
    stratify: bool = True,
) -> tuple:
    """
    Split the processed DataFrame into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Fully encoded and feature-engineered DataFrame.
    target : str
        Name of the target column.
    test_size : float
        Fraction of data reserved for testing.
    stratify : bool
        Whether to stratify by the target class distribution.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    drop_cols = [target]
    if "customerID" in df.columns:
        drop_cols.append("customerID")

    X = df.drop(columns=drop_cols)
    y = df[target]

    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=strat,
    )
    print(
        f"[data_utils] Train size: {len(X_train):,} | Test size: {len(X_test):,} | "
        f"Churn rate train: {y_train.mean():.2%} | Churn rate test: {y_test.mean():.2%}"
    )
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────────────────────
# Versioned Data Saving
# ──────────────────────────────────────────────────────────────────────────────

def save_versioned_dataset(df: pd.DataFrame, output_dir: str, prefix: str = "processed") -> str:
    """
    Save a processed DataFrame as a timestamped CSV for reproducibility.

    Parameters
    ----------
    df : pd.DataFrame
    output_dir : str
        Directory to save into.
    prefix : str
        Filename prefix.

    Returns
    -------
    str
        Full path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"[data_utils] Saved versioned dataset to: {filepath}")
    return filepath
