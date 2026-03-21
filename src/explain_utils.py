"""
explain_utils.py
----------------
Model explainability using SHAP, Partial Dependence Plots (PDP),
and Individual Conditional Expectation (ICE) plots.

All functions return matplotlib Figure objects so they can be displayed
in both notebooks and the Streamlit dashboard without side effects.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("[explain_utils] shap not installed. Explainability plots disabled.")

from sklearn.inspection import PartialDependenceDisplay


# ──────────────────────────────────────────────────────────────────────────────
# SHAP — Global Summary
# ──────────────────────────────────────────────────────────────────────────────

def shap_summary(
    model,
    X: pd.DataFrame,
    model_type: str = "tree",
    max_display: int = 20,
    figsize: tuple = (10, 7),
) -> plt.Figure:
    """
    Generate a SHAP beeswarm summary plot showing global feature impact.

    Parameters
    ----------
    model : fitted model
        Tree-based or linear model compatible with shap Explainer.
    X : pd.DataFrame
        Feature matrix (test set recommended).
    model_type : str
        'tree' for tree-based models, 'linear' for Logistic Regression.
    max_display : int
        Number of top features to display.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap is required for explainability. Install with: pip install shap")

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # For binary classification, shap_values is a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)

    fig, ax = plt.subplots(figsize=figsize)
    plt.sca(ax)
    shap.summary_plot(
        shap_values, X,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    ax.set_title("SHAP Feature Impact — Global Summary", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def shap_bar_plot(
    model,
    X: pd.DataFrame,
    model_type: str = "tree",
    max_display: int = 15,
    figsize: tuple = (9, 6),
) -> plt.Figure:
    """
    Bar chart of mean absolute SHAP values per feature.

    Parameters
    ----------
    model : fitted model
    X : pd.DataFrame
    model_type : str
    max_display : int
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap is required.")

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df_shap = pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs_shap})
    df_shap = df_shap.sort_values("mean_abs_shap", ascending=True).tail(max_display)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(df_shap["feature"], df_shap["mean_abs_shap"], color="#2563EB", alpha=0.85)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("SHAP Feature Importance (Mean Absolute Impact)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# SHAP — Local / Per-Customer Force Plot
# ──────────────────────────────────────────────────────────────────────────────

def shap_force_plot_static(
    model,
    X_single: pd.DataFrame,
    model_type: str = "tree",
    figsize: tuple = (14, 3),
) -> plt.Figure:
    """
    Generate a static waterfall/force plot for a single customer prediction.

    Parameters
    ----------
    model : fitted model
    X_single : pd.DataFrame
        Single-row DataFrame for the customer of interest.
    model_type : str
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap is required.")

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_single)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]
    else:
        explainer = shap.LinearExplainer(model, X_single)
        shap_values = explainer.shap_values(X_single)
        base_value = explainer.expected_value

    fig, ax = plt.subplots(figsize=figsize)
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=X_single.iloc[0].values,
            feature_names=list(X_single.columns),
        ),
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    return fig


def get_customer_shap_contributions(
    model,
    X_single: pd.DataFrame,
    model_type: str = "tree",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return a DataFrame of the top N SHAP feature contributions for a single
    customer, sorted by absolute impact. Useful for actionable recommendations.

    Parameters
    ----------
    model : fitted model
    X_single : pd.DataFrame
        Single-row DataFrame.
    model_type : str
    top_n : int

    Returns
    -------
    pd.DataFrame
        Columns: feature, value, shap_value, direction
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap is required.")

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_single)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        explainer = shap.LinearExplainer(model, X_single)
        shap_values = explainer.shap_values(X_single)

    df = pd.DataFrame({
        "feature": X_single.columns,
        "value": X_single.iloc[0].values,
        "shap_value": shap_values[0],
    })
    df["abs_shap"] = df["shap_value"].abs()
    df["direction"] = df["shap_value"].apply(lambda v: "increases_churn" if v > 0 else "reduces_churn")
    df = df.sort_values("abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    return df[["feature", "value", "shap_value", "direction"]]


# ──────────────────────────────────────────────────────────────────────────────
# Partial Dependence Plots (PDP)
# ──────────────────────────────────────────────────────────────────────────────

def plot_partial_dependence(
    model,
    X: pd.DataFrame,
    features: list,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Plot Partial Dependence for a list of features showing marginal effect
    of each feature on churn probability.

    Parameters
    ----------
    model : fitted model with predict_proba
    X : pd.DataFrame
    features : list
        Feature names (must exist in X.columns).
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    valid_features = [f for f in features if f in X.columns]
    if not valid_features:
        raise ValueError("None of the specified features exist in X.columns.")

    n = len(valid_features)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    display = PartialDependenceDisplay.from_estimator(
        model, X, valid_features,
        kind="average",
        response_method="predict_proba",
        ax=axes,
    )
    fig.suptitle("Partial Dependence Plots", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Individual Conditional Expectation (ICE) Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_ice(
    model,
    X: pd.DataFrame,
    feature: str,
    sample_n: int = 100,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Generate an ICE plot for a single feature, overlaying individual
    customer prediction curves with the average PDP line.

    Parameters
    ----------
    model : fitted model
    X : pd.DataFrame
    feature : str
    sample_n : int
        Number of customers to sample for ICE lines (avoids overplotting).
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found in X.columns.")

    X_sample = X.sample(n=min(sample_n, len(X)), random_state=42)

    fig, ax = plt.subplots(figsize=figsize)
    display = PartialDependenceDisplay.from_estimator(
        model, X_sample, [feature],
        kind="both",
        response_method="predict_proba",
        ice_lines_kw={"color": "#93C5FD", "alpha": 0.3, "linewidth": 0.8},
        pd_line_kw={"color": "#1D4ED8", "linewidth": 2.5, "label": "PDP (mean)"},
        ax=[ax],
    )
    ax.set_title(f"ICE Plot — {feature}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Churn Probability", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig
