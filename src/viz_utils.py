"""
viz_utils.py
------------
All visualization functions for exploratory data analysis, model evaluation,
and business-level reporting in the Telecom Churn pipeline.

Every function returns a matplotlib Figure so it can be embedded in
Streamlit via st.pyplot() or saved to disk for documentation.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.calibration import CalibrationDisplay


# ──────────────────────────────────────────────────────────────────────────────
# Color palette (brand-consistent, professional)
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "primary": "#1D4ED8",
    "secondary": "#64748B",
    "churn": "#DC2626",
    "retain": "#16A34A",
    "neutral": "#E2E8F0",
    "highlight": "#F59E0B",
}


# ──────────────────────────────────────────────────────────────────────────────
# Churn Rate by Categorical Feature
# ──────────────────────────────────────────────────────────────────────────────

def plot_churn_by_feature(
    df: pd.DataFrame,
    feature: str,
    target: str = "Churn",
    figsize: tuple = (8, 4),
    title: str = None,
) -> plt.Figure:
    """
    Bar chart showing churn rate (%) per category of a given feature.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain both feature and target columns in original/decoded form.
    feature : str
    target : str
    figsize : tuple
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = df.copy()
    # Coerce target to numeric (handles 'Yes'/'No' strings from pre-encoded data)
    if df[target].dtype == object or pd.api.types.is_string_dtype(df[target]):
        df[target] = df[target].map({"Yes": 1, "No": 0, "1": 1, "0": 0}).fillna(df[target]).astype(float)

    churn_rate = (
        df.groupby(feature)[target]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )
    churn_rate.columns = [feature, "churn_rate"]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        churn_rate[feature].astype(str),
        churn_rate["churn_rate"],
        color=PALETTE["primary"],
        alpha=0.85,
        width=0.55,
    )

    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Churn Rate (%)", fontsize=10)
    ax.set_xlabel(feature, fontsize=10)
    ax.set_title(title or f"Churn Rate by {feature}", fontsize=12, fontweight="bold")
    ax.set_ylim(0, min(churn_rate["churn_rate"].max() + 12, 100))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Numeric Distribution with Churn Overlay
# ──────────────────────────────────────────────────────────────────────────────

def plot_numeric_distribution(
    df: pd.DataFrame,
    feature: str,
    target: str = "Churn",
    bins: int = 30,
    figsize: tuple = (8, 4),
) -> plt.Figure:
    """
    Overlaid histogram of a numeric feature split by churn status.

    Parameters
    ----------
    df : pd.DataFrame
    feature : str
    target : str
    bins : int
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    df = df.copy()
    if df[target].dtype == object or pd.api.types.is_string_dtype(df[target]):
        df[target] = df[target].map({"Yes": 1, "No": 0}).fillna(df[target]).astype(float)

    for label, color, name in [
        (0, PALETTE["retain"], "Retained"),
        (1, PALETTE["churn"], "Churned"),
    ]:
        subset = df[df[target] == label][feature].dropna()
        ax.hist(
            subset, bins=bins, alpha=0.55,
            color=color, label=name, edgecolor="white", linewidth=0.4,
        )

    ax.set_xlabel(feature, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Distribution of {feature} by Churn Status", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Correlation Heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: list = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Annotated correlation heatmap for numeric features.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list, optional
        Subset of columns to include.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if cols:
        df = df[cols]

    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    cmap = LinearSegmentedColormap.from_list(
        "custom_diverge",
        [PALETTE["churn"], "#FFFFFF", PALETTE["primary"]],
    )

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            if not mask[i, j]:
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if abs(corr.iloc[i, j]) < 0.5 else "white")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list = None,
    figsize: tuple = (5, 4),
) -> plt.Figure:
    """
    Styled confusion matrix.

    Parameters
    ----------
    y_true : pd.Series
    y_pred : np.ndarray
    labels : list, optional
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = labels or ["Retained", "Churned"]
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum() * 100

    fig, ax = plt.subplots(figsize=figsize)
    cmap = LinearSegmentedColormap.from_list("cm_cmap", ["#EFF6FF", PALETTE["primary"]])
    im = ax.imshow(cm, cmap=cmap)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)",
                ha="center", va="center",
                fontsize=11,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontweight="bold",
            )

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# ROC Curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(
    models_results: dict,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """
    Multi-model ROC curve comparison.

    Parameters
    ----------
    models_results : dict
        {model_name: {"y_true": ..., "y_prob": ...}}
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    colors = [PALETTE["primary"], PALETTE["churn"], PALETTE["highlight"],
              PALETTE["retain"], PALETTE["secondary"]]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["neutral"], linewidth=1.2, label="Random classifier")

    for i, (name, result) in enumerate(models_results.items()):
        fpr, tpr, _ = roc_curve(result["y_true"], result["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Precision-Recall Curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_precision_recall_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    model_name: str = "Model",
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """
    Precision-Recall curve for a single model.

    Parameters
    ----------
    y_true : pd.Series
    y_prob : np.ndarray
    model_name : str
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, color=PALETTE["primary"], linewidth=2,
            label=f"{model_name} (AUC = {pr_auc:.3f})")
    ax.axhline(y=baseline, color=PALETTE["secondary"], linestyle="--",
               linewidth=1.2, label=f"Baseline (prevalence = {baseline:.2%})")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Calibration Curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_calibration_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10,
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """
    Reliability / calibration curve to verify predicted probability accuracy.

    Parameters
    ----------
    y_true : pd.Series
    y_prob : np.ndarray
    model_name : str
    n_bins : int
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    CalibrationDisplay.from_predictions(
        y_true, y_prob,
        n_bins=n_bins,
        name=model_name,
        ax=ax,
        color=PALETTE["primary"],
    )
    ax.set_title("Calibration Curve", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Lift / Gain Chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_lift_chart(
    y_true: pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 10,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Cumulative gains / lift chart to evaluate targeting efficiency.
    Shows how much better the model is at identifying churners vs random targeting.

    Parameters
    ----------
    y_true : pd.Series
    y_prob : np.ndarray
    n_bins : int
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = pd.DataFrame({"y_true": y_true.values, "y_prob": y_prob})
    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, q=n_bins, labels=False)

    gains = []
    total_churners = df["y_true"].sum()
    for d in range(n_bins):
        subset = df[df["decile"] <= d]
        gains.append({
            "pct_customers": (d + 1) / n_bins,
            "pct_churners_captured": subset["y_true"].sum() / total_churners,
        })
    gains_df = pd.DataFrame(gains)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Gains chart
    axes[0].plot(gains_df["pct_customers"], gains_df["pct_churners_captured"],
                 color=PALETTE["primary"], linewidth=2, marker="o", markersize=4, label="Model")
    axes[0].plot([0, 1], [0, 1], "--", color=PALETTE["secondary"], linewidth=1.2, label="Random")
    axes[0].set_xlabel("Fraction of Customers Contacted", fontsize=10)
    axes[0].set_ylabel("Fraction of Churners Captured", fontsize=10)
    axes[0].set_title("Cumulative Gains Chart", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Lift chart
    gains_df["lift"] = gains_df["pct_churners_captured"] / gains_df["pct_customers"]
    axes[1].bar(
        range(1, n_bins + 1),
        gains_df["lift"],
        color=PALETTE["primary"], alpha=0.8, width=0.6,
    )
    axes[1].axhline(y=1.0, color=PALETTE["secondary"], linestyle="--", linewidth=1.2, label="No lift")
    axes[1].set_xlabel("Decile", fontsize=10)
    axes[1].set_ylabel("Lift", fontsize=10)
    axes[1].set_title("Lift by Decile", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# High-Risk Customer Table
# ──────────────────────────────────────────────────────────────────────────────

def display_churn_table(
    df_original: pd.DataFrame,
    y_prob: np.ndarray,
    top_n: int = 20,
    id_col: str = "customerID",
) -> pd.DataFrame:
    """
    Build a sorted DataFrame of the highest-risk customers with their
    churn probability and key attributes for export or dashboard display.

    Parameters
    ----------
    df_original : pd.DataFrame
        Original (pre-encoded) DataFrame with customerID and readable features.
    y_prob : np.ndarray
        Predicted churn probabilities aligned with df_original's test rows.
    top_n : int
    id_col : str

    Returns
    -------
    pd.DataFrame
    """
    display_cols = [id_col, "tenure", "MonthlyCharges", "Contract",
                    "InternetService", "network_quality"]
    available = [c for c in display_cols if c in df_original.columns]

    result = df_original[available].copy().reset_index(drop=True)
    result["churn_probability"] = np.round(y_prob, 4)
    result["risk_tier"] = pd.cut(
        result["churn_probability"],
        bins=[0, 0.4, 0.6, 0.8, 1.0],
        labels=["Low", "Medium", "High", "Critical"],
    )
    result = result.sort_values("churn_probability", ascending=False).head(top_n)
    return result.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Word Cloud for Complaint Text
# ──────────────────────────────────────────────────────────────────────────────

def plot_wordcloud(
    texts: pd.Series,
    title: str = "Common Complaint Terms",
    figsize: tuple = (10, 5),
    max_words: int = 80,
    background_color: str = "white",
) -> plt.Figure:
    """
    Generate a word cloud from a Series of cleaned complaint text strings.

    Requires the wordcloud package (pip install wordcloud).
    Falls back to a ranked bar chart of top terms if wordcloud is not installed.

    Parameters
    ----------
    texts : pd.Series
        Series of cleaned text strings (one per customer).
    title : str
    figsize : tuple
    max_words : int
    background_color : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    combined = " ".join(texts.dropna().tolist())

    try:
        from wordcloud import WordCloud
        wc = WordCloud(
            width=900,
            height=400,
            max_words=max_words,
            background_color=background_color,
            colormap="Blues",
            collocations=False,
            prefer_horizontal=0.85,
        ).generate(combined if combined.strip() else "no complaints recorded")

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        plt.tight_layout()
        return fig

    except ImportError:
        # Fallback: frequency bar chart
        from collections import Counter
        words = combined.split()
        counts = Counter(words).most_common(20)
        if not counts:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No complaint text available.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.axis("off")
            return fig

        labels, values = zip(*counts)
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(list(labels)[::-1], list(values)[::-1],
                color=PALETTE["primary"], alpha=0.85)
        ax.set_xlabel("Frequency", fontsize=10)
        ax.set_title(f"{title} (wordcloud not installed — showing top 20 terms)",
                     fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        return fig


def plot_sentiment_distribution(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment_score",
    target: str = "Churn",
    figsize: tuple = (8, 4),
) -> plt.Figure:
    """
    Histogram of sentiment scores split by churn status.

    Parameters
    ----------
    df : pd.DataFrame
    sentiment_col : str
    target : str
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if sentiment_col not in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Sentiment data not available.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    churn_vals = [0, 1]
    colors     = [PALETTE["retain"], PALETTE["churn"]]
    labels_map = {0: "Retained", 1: "Churned"}

    for val, color in zip(churn_vals, colors):
        if target in df.columns:
            subset = df[df[target] == val][sentiment_col].dropna()
        else:
            subset = df[sentiment_col].dropna()

        ax.hist(
            subset, bins=30, alpha=0.55, color=color,
            label=labels_map.get(val, str(val)),
            edgecolor="white", linewidth=0.4,
        )

    ax.axvline(x=0, color=PALETTE["secondary"], linestyle="--", linewidth=1.2, label="Neutral")
    ax.set_xlabel("Sentiment Polarity Score", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Sentiment Score Distribution by Churn Status", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig
