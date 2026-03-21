"""
model_utils.py
--------------
Model training, hyperparameter tuning, evaluation, and persistence
for the Telecom Churn prediction pipeline.

Models covered:
    - Logistic Regression (baseline, interpretable)
    - Random Forest       (non-linear, feature importance)
    - XGBoost             (high-performance, SHAP-compatible)
    - LightGBM            (fast gradient boosting, optional)
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.calibration import CalibratedClassifierCV

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("[model_utils] xgboost not installed. XGBoost training disabled.")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    warnings.warn("[model_utils] lightgbm not installed. LightGBM training disabled.")


RANDOM_STATE = 42
CV_FOLDS = 5


# ──────────────────────────────────────────────────────────────────────────────
# Baseline — Logistic Regression
# ──────────────────────────────────────────────────────────────────────────────

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool = True,
) -> LogisticRegression:
    """
    Train a Logistic Regression classifier with optional hyperparameter tuning.

    Parameters
    ----------
    X_train : pd.DataFrame
        Scaled feature matrix.
    y_train : pd.Series
        Binary target.
    tune : bool
        If True, run RandomizedSearchCV over C and penalty.

    Returns
    -------
    LogisticRegression
        Fitted model.
    """
    if tune:
        param_dist = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
            "class_weight": ["balanced", None],
        }
        base = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        search = RandomizedSearchCV(
            base, param_dist, n_iter=12, cv=cv,
            scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print(f"[model_utils] Logistic Regression best params: {search.best_params_}")
    else:
        model = LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs",
            max_iter=1000, random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)

    print("[model_utils] Logistic Regression training complete.")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Intermediate — Random Forest
# ──────────────────────────────────────────────────────────────────────────────

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool = True,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    tune : bool

    Returns
    -------
    RandomForestClassifier
    """
    if tune:
        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "class_weight": ["balanced", None],
        }
        base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        search = RandomizedSearchCV(
            base, param_dist, n_iter=15, cv=cv,
            scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print(f"[model_utils] Random Forest best params: {search.best_params_}")
    else:
        model = RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(X_train, y_train)

    print("[model_utils] Random Forest training complete.")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# High-Performance — XGBoost
# ──────────────────────────────────────────────────────────────────────────────

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    tune: bool = True,
):
    """
    Train an XGBoost classifier with optional early stopping and tuning.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    X_val : pd.DataFrame, optional
        Validation set for early stopping.
    y_val : pd.Series, optional
    tune : bool

    Returns
    -------
    xgb.XGBClassifier or None
        Returns None if XGBoost is not installed.
    """
    if not XGB_AVAILABLE:
        print("[model_utils] XGBoost not available. Skipping.")
        return None

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    if tune:
        param_dist = {
            "n_estimators": [200, 400, 600],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0, 0.1, 0.3],
            "reg_alpha": [0, 0.1, 1.0],
        }
        base = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        search = RandomizedSearchCV(
            base, param_dist, n_iter=15, cv=cv,
            scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print(f"[model_utils] XGBoost best params: {search.best_params_}")
    else:
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params = {
                "eval_set": [(X_val, y_val)],
                "verbose": False,
            }
        model.fit(X_train, y_train, **fit_params)

    print("[model_utils] XGBoost training complete.")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Optional — LightGBM
# ──────────────────────────────────────────────────────────────────────────────

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool = False,
):
    """
    Train a LightGBM classifier.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    tune : bool

    Returns
    -------
    lgb.LGBMClassifier or None
    """
    if not LGB_AVAILABLE:
        print("[model_utils] LightGBM not available. Skipping.")
        return None

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    print("[model_utils] LightGBM training complete.")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    model_name: str = "Model",
) -> dict:
    """
    Compute a comprehensive evaluation report for a fitted classifier.

    Metrics
    -------
    accuracy, precision, recall, f1, roc_auc, confusion_matrix,
    classification_report, predicted_proba

    Parameters
    ----------
    model : fitted classifier with predict_proba method
    X_test : pd.DataFrame
    y_test : pd.Series
    threshold : float
        Decision threshold for converting probabilities to labels.
    model_name : str
        Label used in print output.

    Returns
    -------
    dict
        All metric values plus raw predictions and probabilities.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "model_name": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    print(
        f"\n[model_utils] {model_name} Evaluation\n"
        f"  Accuracy : {metrics['accuracy']:.4f}\n"
        f"  Precision: {metrics['precision']:.4f}\n"
        f"  Recall   : {metrics['recall']:.4f}\n"
        f"  F1 Score : {metrics['f1']:.4f}\n"
        f"  ROC-AUC  : {metrics['roc_auc']:.4f}"
    )
    return metrics


def get_feature_importance(
    model,
    feature_names: list,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract feature importances from tree-based models or coefficients
    from linear models.

    Parameters
    ----------
    model : fitted model
    feature_names : list
    top_n : int
        Number of top features to return.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with 'feature' and 'importance' columns.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        raise AttributeError("Model does not expose feature importances or coefficients.")

    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return df_imp


def cost_sensitive_evaluation(
    y_test: pd.Series,
    y_prob: np.ndarray,
    cost_fn: float = 500,
    cost_fp: float = 50,
    threshold: float = 0.5,
) -> dict:
    """
    Estimate the financial cost of model errors using domain-specific cost
    assumptions for the Pakistan telecom market.

    Parameters
    ----------
    y_test : pd.Series
        True labels.
    y_prob : np.ndarray
        Predicted probabilities.
    cost_fn : float
        Revenue lost per missed churner (false negative), in USD equivalent.
    cost_fp : float
        Cost of retention offer sent to non-churner (false positive), in USD.
    threshold : float
        Decision threshold.

    Returns
    -------
    dict
        Total cost, breakdown by error type, and cost per customer.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total_cost = (fn * cost_fn) + (fp * cost_fp)
    baseline_cost = y_test.sum() * cost_fn  # Cost if we predicted no churners at all

    result = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "cost_false_negatives": fn * cost_fn,
        "cost_false_positives": fp * cost_fp,
        "total_model_cost": total_cost,
        "baseline_cost_no_model": baseline_cost,
        "cost_saved_vs_baseline": baseline_cost - total_cost,
        "cost_per_customer": total_cost / max(len(y_test), 1),
    }

    print(
        f"\n[model_utils] Cost-Sensitive Evaluation\n"
        f"  False Negative cost: ${result['cost_false_negatives']:,.0f}\n"
        f"  False Positive cost: ${result['cost_false_positives']:,.0f}\n"
        f"  Total model cost   : ${total_cost:,.0f}\n"
        f"  Saved vs baseline  : ${result['cost_saved_vs_baseline']:,.0f}"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Model Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_model(model, filepath: str) -> None:
    """
    Serialize and save a fitted model to disk using pickle.

    Parameters
    ----------
    model : fitted model object
    filepath : str
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"[model_utils] Model saved to: {filepath}")


def load_model(filepath: str):
    """
    Load a pickled model from disk.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    Fitted model object.

    Raises
    ------
    FileNotFoundError
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print(f"[model_utils] Model loaded from: {filepath}")
    return model
