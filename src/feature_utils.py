"""
feature_utils.py
----------------
All feature engineering logic for the Telecom Churn pipeline.

Covers two categories:
    Structured  — network quality, interactions, bundles, tenure brackets,
                  average charges, contract change flag
    Unstructured — complaint text cleaning, TF-IDF vectorisation,
                   sentiment scoring, keyword extraction

Must be called BEFORE encode_categorical() in data_utils.
"""

import re
import string
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    warnings.warn(
        "[feature_utils] textblob not installed. "
        "Sentiment scoring will default to 0. Install with: pip install textblob"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

REGION_NETWORK_QUALITY_MAP = {
    "DSL": 1,
    "Fiber optic": 2,
    "No": 0,
}

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

TENURE_BINS   = [0, 12, 24, 36, 48, 60, np.inf]
TENURE_LABELS = ["0-12", "13-24", "25-36", "37-48", "49-60", "60+"]

NETWORK_KEYWORDS = [
    "network", "signal", "coverage", "dropped", "slow", "speed",
    "4g", "3g", "disconnected", "outage", "weak",
]
BILLING_KEYWORDS = [
    "bill", "charge", "overcharged", "refund", "price",
    "expensive", "extra", "payment", "deducted", "balance",
]
SERVICE_KEYWORDS = [
    "service", "support", "help", "response", "staff",
    "wait", "rude", "resolved", "complaint",
]

CUSTOM_STOPWORDS = {
    "hai", "ka", "ki", "ke", "ko", "se", "mein", "aur", "bhi",
    "yeh", "woh", "hain", "nahi", "nhi", "kya", "tha", "thi",
    "the", "a", "an", "is", "it", "in", "of", "to", "for",
    "on", "at", "by", "or", "and", "but", "not", "with",
}


# ──────────────────────────────────────────────────────────────────────────────
# Structured: Network Quality
# ──────────────────────────────────────────────────────────────────────────────

def add_network_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign an ordinal network quality score (0=Low, 1=Medium, 2=High).

    If the column already exists in the uploaded CSV, it is preserved as-is.
    Otherwise it is derived from the InternetService column.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    if "network_quality" in df.columns:
        df["network_quality"] = (
            pd.to_numeric(df["network_quality"], errors="coerce").fillna(0).astype(int)
        )
        print("[feature_utils] 'network_quality' found in data — used as-is.")
        return df

    if "InternetService" not in df.columns:
        df["network_quality"] = 0
        print("[feature_utils] Warning: InternetService missing; network_quality set to 0.")
        return df

    df["network_quality"] = (
        df["InternetService"].map(REGION_NETWORK_QUALITY_MAP).fillna(0).astype(int)
    )
    print("[feature_utils] Added 'network_quality' from InternetService.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Structured: Interaction Features
# ──────────────────────────────────────────────────────────────────────────────

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplicative interaction terms: tenure x network_quality
    and MonthlyCharges x network_quality.
    """
    df = df.copy()
    required = {"tenure", "MonthlyCharges", "network_quality"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for interaction features: {missing}")

    df["tenure_network_interaction"]  = df["tenure"] * df["network_quality"]
    df["charges_network_interaction"] = df["MonthlyCharges"] * df["network_quality"]
    print("[feature_utils] Added network interaction features.")
    return df


def add_polynomial_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cols: list,
    degree: int = 2,
) -> tuple:
    """
    Expand numeric columns to polynomial interaction terms for Logistic Regression.

    Returns
    -------
    tuple : (X_train_poly, X_test_poly)
    """
    poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
    train_poly = poly.fit_transform(X_train[cols])
    test_poly  = poly.transform(X_test[cols])

    feature_names  = poly.get_feature_names_out(cols)
    train_poly_df  = pd.DataFrame(train_poly, columns=feature_names, index=X_train.index)
    test_poly_df   = pd.DataFrame(test_poly,  columns=feature_names, index=X_test.index)

    X_train_out = pd.concat([X_train.drop(columns=cols), train_poly_df], axis=1)
    X_test_out  = pd.concat([X_test.drop(columns=cols),  test_poly_df],  axis=1)

    print(f"[feature_utils] Polynomial features (degree={degree}) added for: {cols}")
    return X_train_out, X_test_out


# ──────────────────────────────────────────────────────────────────────────────
# Structured: Service Bundle Features
# ──────────────────────────────────────────────────────────────────────────────

def derive_bundle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count subscribed services (0-8) and flag high-value customers.

    Features added
    --------------
    service_bundle_count : int
    high_value_customer  : int
    """
    df = df.copy()
    service_cols_present = [c for c in SERVICE_COLUMNS if c in df.columns]

    if service_cols_present:
        # Use is_numeric_dtype to handle both object and pandas StringDtype (pandas 2.x)
        first_col = df[service_cols_present[0]]
        if pd.api.types.is_numeric_dtype(first_col):
            df["service_bundle_count"] = df[service_cols_present].sum(axis=1).astype(int)
        else:
            df["service_bundle_count"] = df[service_cols_present].apply(
                lambda row: (row == "Yes").sum(), axis=1
            ).astype(int)
    else:
        df["service_bundle_count"] = 0
        print("[feature_utils] Warning: No service columns found; service_bundle_count set to 0.")

    if "MonthlyCharges" in df.columns:
        threshold = df["MonthlyCharges"].quantile(0.75)
        df["high_value_customer"] = (df["MonthlyCharges"] > threshold).astype(int)
    else:
        df["high_value_customer"] = 0

    print("[feature_utils] Added 'service_bundle_count' and 'high_value_customer'.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Structured: Tenure and Charge Derived Features
# ──────────────────────────────────────────────────────────────────────────────

def add_tenure_bracket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin tenure into categorical brackets.

    Features added
    --------------
    tenure_bracket         : category label
    tenure_bracket_encoded : int ordinal
    """
    df = df.copy()
    df["tenure_bracket"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, right=True,
    )
    label_order = {label: i for i, label in enumerate(TENURE_LABELS)}
    df["tenure_bracket_encoded"] = df["tenure_bracket"].map(label_order).astype(int)
    print("[feature_utils] Added 'tenure_bracket' and 'tenure_bracket_encoded'.")
    return df


def add_avg_monthly_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TotalCharges / tenure. Falls back to MonthlyCharges when tenure == 0.

    Feature added
    -------------
    avg_monthly_charges : float
    """
    df = df.copy()
    if "TotalCharges" not in df.columns:
        df["avg_monthly_charges"] = df.get("MonthlyCharges", 0)
        return df

    with np.errstate(divide="ignore", invalid="ignore"):
        df["avg_monthly_charges"] = np.where(
            df["tenure"] > 0,
            df["TotalCharges"] / df["tenure"],
            df.get("MonthlyCharges", 0),
        )
    print("[feature_utils] Added 'avg_monthly_charges'.")
    return df


def flag_recent_contract_change(df: pd.DataFrame, tenure_threshold: int = 3) -> pd.DataFrame:
    """
    Flag month-to-month customers with tenure <= threshold as recently changed.

    Feature added
    -------------
    recent_contract_change : int (0/1)
    """
    df = df.copy()
    if "Contract" in df.columns and df["Contract"].dtype == object:
        df["recent_contract_change"] = (
            (df["tenure"] <= tenure_threshold) & (df["Contract"] == "Month-to-month")
        ).astype(int)
    else:
        df["recent_contract_change"] = 0
    print("[feature_utils] Added 'recent_contract_change'.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Unstructured: Text Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean a single complaint / feedback string for NLP processing.

    Steps
    -----
    1. Lowercase
    2. Remove URLs, @ mentions
    3. Strip punctuation and digits
    4. Collapse whitespace
    5. Remove English + Roman Urdu stopwords

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Cleaned text ready for TF-IDF or sentiment scoring.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [w for w in text.split() if w not in CUSTOM_STOPWORDS and len(w) > 1]
    return " ".join(tokens)


def preprocess_text_column(df: pd.DataFrame, text_col: str = "complaint_text") -> pd.DataFrame:
    """
    Apply clean_text to every row of the complaint column.

    Feature added
    -------------
    complaint_text_clean : str

    Parameters
    ----------
    df : pd.DataFrame
    text_col : str
        Source column name in the DataFrame.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    if text_col not in df.columns:
        df["complaint_text_clean"] = ""
        print(f"[feature_utils] Column '{text_col}' not found; complaint_text_clean set to empty.")
        return df

    df["complaint_text_clean"] = df[text_col].fillna("").apply(clean_text)
    print(f"[feature_utils] Text cleaned into 'complaint_text_clean'.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Unstructured: Sentiment Scoring
# ──────────────────────────────────────────────────────────────────────────────

def _textblob_polarity(text: str) -> float:
    """Return TextBlob polarity [-1, 1], or 0.0 on failure."""
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0


def add_sentiment_score(df: pd.DataFrame, text_col: str = "complaint_text_clean") -> pd.DataFrame:
    """
    Compute per-customer sentiment polarity from complaint text.

    Features added
    --------------
    sentiment_score    : float in [-1, 1]
    sentiment_negative : int  (1 if score < -0.1)
    sentiment_positive : int  (1 if score >  0.1)

    Parameters
    ----------
    df : pd.DataFrame
    text_col : str
        Cleaned text column produced by preprocess_text_column.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    if text_col not in df.columns or not TEXTBLOB_AVAILABLE:
        df["sentiment_score"]    = 0.0
        df["sentiment_negative"] = 0
        df["sentiment_positive"] = 0
        if not TEXTBLOB_AVAILABLE:
            print("[feature_utils] TextBlob unavailable; sentiment features set to 0.")
        return df

    df["sentiment_score"]    = df[text_col].apply(_textblob_polarity)
    df["sentiment_negative"] = (df["sentiment_score"] < -0.1).astype(int)
    df["sentiment_positive"] = (df["sentiment_score"] >  0.1).astype(int)
    print("[feature_utils] Added sentiment features.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Unstructured: Keyword Flags
# ──────────────────────────────────────────────────────────────────────────────

def add_keyword_flags(df: pd.DataFrame, text_col: str = "complaint_text_clean") -> pd.DataFrame:
    """
    Binary flags indicating complaint category: network, billing, or service.

    Features added
    --------------
    complaint_network : int
    complaint_billing : int
    complaint_service : int

    Parameters
    ----------
    df : pd.DataFrame
    text_col : str

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    if text_col not in df.columns:
        df["complaint_network"] = 0
        df["complaint_billing"] = 0
        df["complaint_service"] = 0
        return df

    def _has(text: str, keywords: list) -> int:
        if not text:
            return 0
        return int(any(kw in text for kw in keywords))

    df["complaint_network"] = df[text_col].apply(lambda t: _has(t, NETWORK_KEYWORDS))
    df["complaint_billing"] = df[text_col].apply(lambda t: _has(t, BILLING_KEYWORDS))
    df["complaint_service"] = df[text_col].apply(lambda t: _has(t, SERVICE_KEYWORDS))

    print("[feature_utils] Added complaint keyword flags (network, billing, service).")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Unstructured: TF-IDF Vectorisation
# ──────────────────────────────────────────────────────────────────────────────

def build_tfidf_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    text_col: str = "complaint_text_clean",
    max_features: int = 50,
    ngram_range: tuple = (1, 2),
) -> tuple:
    """
    Fit TF-IDF on train text and append sparse columns to both splits.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    text_col : str
    max_features : int
    ngram_range : tuple

    Returns
    -------
    tuple : (X_train_out, X_test_out, fitted_vectorizer)
        vectorizer is None if text_col is missing.
    """
    if text_col not in X_train.columns:
        print(f"[feature_utils] '{text_col}' not found; skipping TF-IDF.")
        return X_train, X_test, None

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        sublinear_tf=True,
    )

    train_texts = X_train[text_col].fillna("").tolist()
    test_texts  = X_test[text_col].fillna("").tolist()

    train_tfidf = vectorizer.fit_transform(train_texts).toarray()
    test_tfidf  = vectorizer.transform(test_texts).toarray()

    col_names      = [f"tfidf_{t}" for t in vectorizer.get_feature_names_out()]
    train_tfidf_df = pd.DataFrame(train_tfidf, columns=col_names, index=X_train.index)
    test_tfidf_df  = pd.DataFrame(test_tfidf,  columns=col_names, index=X_test.index)

    X_train_out = pd.concat([X_train.drop(columns=[text_col]), train_tfidf_df], axis=1)
    X_test_out  = pd.concat([X_test.drop(columns=[text_col]),  test_tfidf_df],  axis=1)

    print(f"[feature_utils] TF-IDF: {max_features} features appended.")
    return X_train_out, X_test_out, vectorizer


def get_tfidf_top_terms(vectorizer: TfidfVectorizer, top_n: int = 20) -> pd.DataFrame:
    """
    Return top n terms by lowest IDF (most frequent across documents).

    Parameters
    ----------
    vectorizer : fitted TfidfVectorizer
    top_n : int

    Returns
    -------
    pd.DataFrame : columns [term, idf_score]
    """
    if vectorizer is None:
        return pd.DataFrame(columns=["term", "idf_score"])

    terms = vectorizer.get_feature_names_out()
    idf   = vectorizer.idf_
    df_t  = pd.DataFrame({"term": terms, "idf_score": idf})
    return df_t.sort_values("idf_score").head(top_n).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Master Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full structured + unstructured feature engineering pipeline.

    Structured steps (always run):
        1. Network quality
        2. Interaction features
        3. Bundle features
        4. Tenure bracket
        5. Average monthly charges
        6. Recent contract change flag

    Unstructured steps (run only if 'complaint_text' column is present):
        7. Text cleaning
        8. Sentiment scoring
        9. Keyword flags

    Note
    ----
    TF-IDF vectorisation is NOT performed here because it requires a
    train/test split. Call build_tfidf_features() after split_train_test().

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame; must be called BEFORE encode_categorical().

    Returns
    -------
    pd.DataFrame
    """
    # Structured
    df = add_network_quality(df)
    df = create_interaction_features(df)
    df = derive_bundle_features(df)
    df = add_tenure_bracket(df)
    df = add_avg_monthly_charges(df)
    df = flag_recent_contract_change(df)

    # Unstructured
    df = preprocess_text_column(df, text_col="complaint_text")
    df = add_sentiment_score(df, text_col="complaint_text_clean")
    df = add_keyword_flags(df, text_col="complaint_text_clean")

    print("[feature_utils] Full feature engineering pipeline complete.")
    return df
