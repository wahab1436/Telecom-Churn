"""
app.py
------
Streamlit dashboard — Telecom Customer Churn Intelligence.

New in this version (Blueprint v2):
    - CSV upload from PC via st.file_uploader
    - Data preview after upload
    - NLP: complaint text cleaning, sentiment scoring, keyword flags
    - Word cloud for common complaint terms
    - Sidebar sentiment range filter
    - Churn analysis by sentiment and complaint category
    - All existing evaluation, SHAP, cost, and download features retained

Run with:
    streamlit run app.py
"""

import os
import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Project Module Imports
# ──────────────────────────────────────────────────────────────────────────────

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_utils import (
    load_data, load_from_upload, validate_uploaded_columns,
    clean_total_charges, remove_duplicates,
    encode_categorical, split_train_test, scale_numeric_features,
)
from feature_utils import run_feature_engineering, build_tfidf_features
from model_utils import (
    train_logistic_regression, train_random_forest, train_xgboost,
    evaluate_model, get_feature_importance, save_model, load_model,
    cost_sensitive_evaluation,
)
from viz_utils import (
    plot_churn_by_feature, plot_numeric_distribution,
    plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_calibration_curve,
    plot_lift_chart, display_churn_table,
    plot_wordcloud, plot_sentiment_distribution,
)

try:
    from explain_utils import shap_bar_plot, get_customer_shap_contributions
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Anchor all paths to the directory that contains app.py so the dashboard
# works correctly regardless of the working directory from which
# `streamlit run app.py` is invoked.
APP_DIR           = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(APP_DIR, "data", "telco_churn.csv")
MODEL_DIR         = os.path.join(APP_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

NETWORK_LABELS = {0: "Low", 1: "Medium", 2: "High"}


# ──────────────────────────────────────────────────────────────────────────────
# Data Pipeline (cached per unique upload)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def prepare_data(df_raw: pd.DataFrame):
    """
    Run the full cleaning + feature engineering + encoding + split pipeline
    on any DataFrame (uploaded or loaded from disk).

    Returns
    -------
    tuple : (df_display, df_enc, X_train, X_test, y_train, y_test,
             X_train_sc, X_test_sc, scaler, has_target, has_text)
    """
    has_target = "Churn" in df_raw.columns
    has_text   = "complaint_text" in df_raw.columns

    df = clean_total_charges(df_raw.copy())
    df = remove_duplicates(df)
    df = run_feature_engineering(df)     # structured + unstructured features

    df_display = df.copy()               # pre-encoded copy for display

    # Drop text columns before encoding — they cannot be one-hot encoded
    text_cols_to_drop = [c for c in ["complaint_text", "complaint_text_clean"]
                         if c in df.columns]

    if has_target:
        df_enc = encode_categorical(df.drop(columns=text_cols_to_drop, errors="ignore"))
        X_train, X_test, y_train, y_test = split_train_test(df_enc)
        X_train_sc, X_test_sc, scaler    = scale_numeric_features(X_train.copy(), X_test.copy())
    else:
        # Inference-only upload: no target column
        df_enc     = encode_categorical(df.drop(columns=text_cols_to_drop + ["Churn"],
                                                errors="ignore"))
        X_train    = df_enc
        X_test     = df_enc
        y_train    = pd.Series(dtype=int)
        y_test     = pd.Series(dtype=int)
        X_train_sc = df_enc
        X_test_sc  = df_enc
        scaler     = None

    return (df_display, df_enc,
            X_train, X_test, y_train, y_test,
            X_train_sc, X_test_sc, scaler,
            has_target, has_text)


@st.cache_resource(show_spinner=False)
def train_or_load_models(_X_train, _y_train, _X_test, _y_test, _X_train_sc, _X_test_sc):
    """Train or reload all models. Prefixed args with _ to avoid hash issues."""
    results = {}

    # Logistic Regression
    lr_path = os.path.join(MODEL_DIR, "logistic_regression.pkl")
    lr = load_model(lr_path) if os.path.exists(lr_path) else \
         train_logistic_regression(_X_train_sc, _y_train, tune=False)
    if not os.path.exists(lr_path):
        save_model(lr, lr_path)
    results["Logistic Regression"] = {
        "model": lr,
        "eval": evaluate_model(lr, _X_test_sc, _y_test, model_name="Logistic Regression"),
        "X_test": _X_test_sc,
    }

    # Random Forest
    rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    rf = load_model(rf_path) if os.path.exists(rf_path) else \
         train_random_forest(_X_train, _y_train, tune=False)
    if not os.path.exists(rf_path):
        save_model(rf, rf_path)
    results["Random Forest"] = {
        "model": rf,
        "eval": evaluate_model(rf, _X_test, _y_test, model_name="Random Forest"),
        "X_test": _X_test,
    }

    # XGBoost
    xgb_path  = os.path.join(MODEL_DIR, "xgboost.pkl")
    xgb_model = None
    if os.path.exists(xgb_path):
        try:
            xgb_model = load_model(xgb_path)
        except Exception:
            xgb_model = None
    else:
        xgb_model = train_xgboost(_X_train, _y_train, tune=False)
        if xgb_model:
            save_model(xgb_model, xgb_path)

    if xgb_model:
        results["XGBoost"] = {
            "model": xgb_model,
            "eval": evaluate_model(xgb_model, _X_test, _y_test, model_name="XGBoost"),
            "X_test": _X_test,
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar(df_display: pd.DataFrame, has_text: bool) -> dict:
    """Build sidebar filters and return selected values as a dict."""
    st.sidebar.title("Filters")
    st.sidebar.markdown("---")

    # Network quality
    nq_options  = sorted(df_display["network_quality"].unique().tolist())
    selected_nq = st.sidebar.multiselect(
        "Network Quality",
        options=nq_options,
        default=nq_options,
        format_func=lambda v: NETWORK_LABELS.get(v, str(v)),
    )

    # Tenure
    t_min, t_max = int(df_display["tenure"].min()), int(df_display["tenure"].max())
    tenure_range = st.sidebar.slider("Tenure (months)", t_min, t_max, (t_min, t_max))

    # Monthly charges
    c_min = float(df_display["MonthlyCharges"].min())
    c_max = float(df_display["MonthlyCharges"].max())
    charge_range = st.sidebar.slider(
        "Monthly Charges (USD)", c_min, c_max, (c_min, c_max), step=0.5,
    )

    # Sentiment filter — only shown when complaint_text is present
    sentiment_range = (-1.0, 1.0)
    if has_text and "sentiment_score" in df_display.columns:
        sentiment_range = st.sidebar.slider(
            "Sentiment Score Range",
            min_value=-1.0,
            max_value=1.0,
            value=(-1.0, 1.0),
            step=0.05,
            help="Filter by customer sentiment polarity. Negative = dissatisfied.",
        )

    # Decision threshold
    threshold = st.sidebar.slider(
        "Churn Decision Threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Probability cutoff for churn classification.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Telecom Churn Intelligence v2.0")

    return {
        "network_quality":  selected_nq,
        "tenure_range":     tenure_range,
        "charge_range":     charge_range,
        "sentiment_range":  sentiment_range,
        "threshold":        threshold,
    }


def filter_customers(df: pd.DataFrame, filters: dict, has_text: bool) -> pd.DataFrame:
    """Apply sidebar filters to the display DataFrame."""
    mask = (
        df["network_quality"].isin(filters["network_quality"]) &
        df["tenure"].between(*filters["tenure_range"]) &
        df["MonthlyCharges"].between(*filters["charge_range"])
    )
    if has_text and "sentiment_score" in df.columns:
        lo, hi = filters["sentiment_range"]
        mask = mask & df["sentiment_score"].between(lo, hi)

    return df[mask].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# KPI Cards
# ──────────────────────────────────────────────────────────────────────────────

def display_metrics(metrics: dict) -> None:
    """Render model KPI cards in a 5-column row."""
    cols = st.columns(5)
    kpis = [
        ("Accuracy",  f"{metrics['accuracy']:.3f}"),
        ("Precision", f"{metrics['precision']:.3f}"),
        ("Recall",    f"{metrics['recall']:.3f}"),
        ("F1 Score",  f"{metrics['f1']:.3f}"),
        ("ROC-AUC",   f"{metrics['roc_auc']:.3f}"),
    ]
    for col, (label, value) in zip(cols, kpis):
        with col:
            st.metric(label=label, value=value)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():

    # ── Header ──────────────────────────────────────────────────────────────
    st.title("Telecom Customer Churn Intelligence")
    st.markdown(
        "Upload your customer dataset to identify high-risk churners, "
        "analyse complaint patterns, and understand model decisions. "
        "Pakistan telecom market context."
    )
    st.markdown("---")

    # ── Data Source Selection ────────────────────────────────────────────────
    st.subheader("Data Source")

    source_mode = st.radio(
        "Select data source",
        options=["Upload CSV from PC", "Use default dataset (data/telco_churn.csv)"],
        horizontal=True,
    )

    df_raw = None

    if source_mode == "Upload CSV from PC":
        uploaded_file = st.file_uploader(
            "Upload your customer CSV file",
            type=["csv"],
            help=(
                "Required columns: tenure, MonthlyCharges. "
                "Optional: Churn (for model evaluation), complaint_text (for NLP analysis), "
                "network_quality (0/1/2), InternetService, Contract, and standard Telco columns."
            ),
        )

        if uploaded_file is not None:
            try:
                df_raw = load_from_upload(uploaded_file)
                missing_cols = validate_uploaded_columns(df_raw)
                if missing_cols:
                    st.error(
                        f"Uploaded file is missing required columns: {missing_cols}. "
                        "Please check your CSV and re-upload."
                    )
                    df_raw = None
                else:
                    st.success(
                        f"File loaded successfully: {len(df_raw):,} rows, "
                        f"{df_raw.shape[1]} columns."
                    )
                    with st.expander("Data Preview (first 10 rows)"):
                        st.dataframe(df_raw.head(10), use_container_width=True)
            except ValueError as e:
                st.error(str(e))
                df_raw = None
        else:
            st.info("Upload a CSV file to begin.")

    else:
        if os.path.exists(DEFAULT_DATA_PATH):
            df_raw = load_data(DEFAULT_DATA_PATH)
            st.success(f"Default dataset loaded: {len(df_raw):,} rows.")
        else:
            st.error(
                f"Default dataset not found at `{DEFAULT_DATA_PATH}`. "
                "Place `telco_churn.csv` in the `data/` directory or upload a file."
            )

    if df_raw is None:
        st.stop()

    st.markdown("---")

    # ── Pipeline ─────────────────────────────────────────────────────────────
    with st.spinner("Processing data and engineering features..."):
        (
            df_display, df_enc,
            X_train, X_test, y_train, y_test,
            X_train_sc, X_test_sc, scaler,
            has_target, has_text,
        ) = prepare_data(df_raw)

    if has_text:
        st.info(
            "Complaint text column detected. "
            "Sentiment scoring and NLP analysis are enabled."
        )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    filters     = render_sidebar(df_display, has_text)
    df_filtered = filter_customers(df_display, filters, has_text)
    st.caption(f"Showing {len(df_filtered):,} of {len(df_display):,} customers after filters.")

    # ── NLP Section (only if complaint_text present) ─────────────────────────
    if has_text:
        st.subheader("Complaint Text Analysis")

        col_wc, col_sent = st.columns(2)

        with col_wc:
            st.markdown("**Word Cloud — Common Complaint Terms**")
            churned_text = df_filtered.get("complaint_text_clean", pd.Series(dtype=str))
            if "Churn" in df_filtered.columns:
                churned_only = df_filtered[df_filtered["Churn"] == 1]
                churned_text = churned_only.get("complaint_text_clean", pd.Series(dtype=str))
            try:
                fig_wc = plot_wordcloud(churned_text, title="Churned Customer Complaints")
                st.pyplot(fig_wc, use_container_width=True)
                plt.close(fig_wc)
            except Exception as e:
                st.warning(f"Word cloud unavailable: {e}")

        with col_sent:
            st.markdown("**Sentiment Score Distribution**")
            try:
                fig_sent = plot_sentiment_distribution(df_filtered)
                st.pyplot(fig_sent, use_container_width=True)
                plt.close(fig_sent)
            except Exception as e:
                st.warning(f"Sentiment plot unavailable: {e}")

        # Complaint category breakdown
        st.markdown("**Complaint Category Breakdown**")
        for flag_col, label in [
            ("complaint_network", "Network Complaints"),
            ("complaint_billing", "Billing Complaints"),
            ("complaint_service", "Service Complaints"),
        ]:
            if flag_col in df_filtered.columns:
                count = df_filtered[flag_col].sum()
                pct   = count / max(len(df_filtered), 1) * 100
                st.markdown(f"- {label}: **{count:,}** customers ({pct:.1f}%)")

        # Churn by sentiment tier
        if "Churn" in df_filtered.columns and "sentiment_negative" in df_filtered.columns:
            st.markdown("**Churn Rate: Negative vs Non-Negative Sentiment**")
            try:
                fig_senti_churn = plot_churn_by_feature(
                    df_filtered, "sentiment_negative",
                    title="Churn Rate — Negative Sentiment Flag",
                )
                st.pyplot(fig_senti_churn, use_container_width=True)
                plt.close(fig_senti_churn)
            except Exception as e:
                st.warning(f"Could not plot: {e}")

        st.markdown("---")

    # ── Model Training ────────────────────────────────────────────────────────
    if not has_target:
        st.warning(
            "No 'Churn' column found in the uploaded data. "
            "Model evaluation metrics are unavailable. "
            "Upload a CSV with a Churn column (Yes/No or 1/0) to enable evaluation."
        )
        st.stop()

    with st.spinner("Training or loading models..."):
        model_results = train_or_load_models(
            X_train, y_train, X_test, y_test, X_train_sc, X_test_sc,
        )

    # ── Model Selector ────────────────────────────────────────────────────────
    st.subheader("Model Selection")
    selected_model_name = st.selectbox(
        "Select model for evaluation and explainability",
        options=list(model_results.keys()),
    )
    selected     = model_results[selected_model_name]
    eval_result  = selected["eval"]
    model        = selected["model"]
    X_test_model = selected["X_test"]
    threshold    = filters["threshold"]
    y_prob       = eval_result["y_prob"]
    y_pred_thresh = (y_prob >= threshold).astype(int)

    # ── KPI Metrics ──────────────────────────────────────────────────────────
    st.subheader("Model Performance")
    display_metrics(eval_result)

    if threshold != 0.5:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        st.markdown(f"Metrics recalculated at threshold **{threshold:.2f}**")
        display_metrics({
            "accuracy":  accuracy_score(y_test, y_pred_thresh),
            "precision": precision_score(y_test, y_pred_thresh, zero_division=0),
            "recall":    recall_score(y_test, y_pred_thresh, zero_division=0),
            "f1":        f1_score(y_test, y_pred_thresh, zero_division=0),
            "roc_auc":   eval_result["roc_auc"],
        })

    st.markdown("---")

    # ── Evaluation Plots ──────────────────────────────────────────────────────
    st.subheader("Evaluation Diagnostics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Confusion Matrix**")
        fig_cm = plot_confusion_matrix(y_test, y_pred_thresh)
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    with col2:
        st.markdown("**ROC Curve**")
        roc_data = {
            name: {"y_true": y_test.values, "y_prob": res["eval"]["y_prob"]}
            for name, res in model_results.items()
        }
        fig_roc = plot_roc_curve(roc_data)
        st.pyplot(fig_roc, use_container_width=True)
        plt.close(fig_roc)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Precision-Recall Curve**")
        fig_pr = plot_precision_recall_curve(y_test, y_prob, model_name=selected_model_name)
        st.pyplot(fig_pr, use_container_width=True)
        plt.close(fig_pr)

    with col4:
        st.markdown("**Calibration Curve**")
        fig_cal = plot_calibration_curve(y_test, y_prob, model_name=selected_model_name)
        st.pyplot(fig_cal, use_container_width=True)
        plt.close(fig_cal)

    st.markdown("**Lift / Gain Chart**")
    fig_lift = plot_lift_chart(y_test, y_prob)
    st.pyplot(fig_lift, use_container_width=True)
    plt.close(fig_lift)

    st.markdown("---")

    # ── Financial Impact ──────────────────────────────────────────────────────
    st.subheader("Financial Impact Estimation")
    cost_col1, cost_col2 = st.columns(2)
    with cost_col1:
        cost_fn = st.number_input("Cost per missed churner (USD)", value=500, min_value=0)
    with cost_col2:
        cost_fp = st.number_input("Cost per false retention offer (USD)", value=50, min_value=0)

    cost_result = cost_sensitive_evaluation(y_test, y_prob, cost_fn, cost_fp, threshold)
    c1, c2, c3  = st.columns(3)
    c1.metric("Total Model Cost",         f"${cost_result['total_model_cost']:,.0f}")
    c2.metric("Baseline Cost (no model)", f"${cost_result['baseline_cost_no_model']:,.0f}")
    c3.metric("Savings vs Baseline",      f"${cost_result['cost_saved_vs_baseline']:,.0f}")

    st.markdown("---")

    # ── Churn by Business Segment ─────────────────────────────────────────────
    st.subheader("Churn Analysis by Business Segment")

    seg_features = [
        col for col in ["network_quality", "Contract", "tenure_bracket",
                        "InternetService", "PaymentMethod",
                        "complaint_network", "complaint_billing", "complaint_service"]
        if col in df_filtered.columns and "Churn" in df_filtered.columns
    ]

    if seg_features:
        tabs = st.tabs(seg_features)
        for tab, feat in zip(tabs, seg_features):
            with tab:
                try:
                    fig_seg = plot_churn_by_feature(df_filtered, feat)
                    st.pyplot(fig_seg, use_container_width=True)
                    plt.close(fig_seg)
                except Exception as e:
                    st.warning(f"Cannot plot {feat}: {e}")

    st.markdown("---")

    # ── Numeric Distributions ─────────────────────────────────────────────────
    st.subheader("Numeric Feature Distributions")
    numeric_options = [
        c for c in ["tenure", "MonthlyCharges", "TotalCharges",
                    "avg_monthly_charges", "service_bundle_count",
                    "network_quality", "sentiment_score"]
        if c in df_filtered.columns
    ]
    if numeric_options:
        selected_num = st.selectbox("Select feature", numeric_options)
        if "Churn" in df_filtered.columns:
            fig_dist = plot_numeric_distribution(df_filtered, selected_num)
            st.pyplot(fig_dist, use_container_width=True)
            plt.close(fig_dist)

    st.markdown("---")

    # ── Feature Importance ────────────────────────────────────────────────────
    st.subheader("Feature Importance")
    try:
        fi_df = get_feature_importance(model, list(X_test_model.columns), top_n=20)
        fig_fi, ax = plt.subplots(figsize=(8, 6))
        ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1],
                color="#1D4ED8", alpha=0.85)
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(f"Top 20 Features — {selected_model_name}", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_fi, use_container_width=True)
        plt.close(fig_fi)
    except Exception as e:
        st.warning(f"Feature importance unavailable: {e}")

    if SHAP_AVAILABLE and selected_model_name in ("Random Forest", "XGBoost"):
        st.markdown("**SHAP Global Feature Impact**")
        with st.spinner("Computing SHAP values..."):
            try:
                X_shap   = X_test_model.sample(n=min(200, len(X_test_model)), random_state=42)
                fig_shap = shap_bar_plot(model, X_shap, model_type="tree", max_display=15)
                st.pyplot(fig_shap, use_container_width=True)
                plt.close(fig_shap)
            except Exception as e:
                st.warning(f"SHAP bar plot unavailable: {e}")

    st.markdown("---")

    # ── Per-Customer SHAP ─────────────────────────────────────────────────────
    if SHAP_AVAILABLE and selected_model_name in ("Random Forest", "XGBoost"):
        st.subheader("Per-Customer Churn Explanation")
        cust_idx = st.number_input(
            "Customer index (test set)",
            min_value=0, max_value=len(X_test_model) - 1, value=0, step=1,
        )
        if st.button("Explain This Customer"):
            with st.spinner("Computing SHAP contributions..."):
                try:
                    X_cust   = X_test_model.iloc[[cust_idx]]
                    contrib  = get_customer_shap_contributions(model, X_cust, top_n=10)
                    prob     = model.predict_proba(X_cust)[0, 1]
                    st.markdown(f"**Predicted churn probability: {prob:.2%}**")
                    st.dataframe(contrib, use_container_width=True)
                except Exception as e:
                    st.warning(f"Per-customer SHAP unavailable: {e}")

    st.markdown("---")

    # ── High-Risk Customer Table ──────────────────────────────────────────────
    st.subheader("High-Risk Customer Roster")
    top_n_customers = st.slider("Customers to display", 10, 100, 25)

    test_display = df_display.loc[y_test.index].reset_index(drop=True)
    risk_table   = display_churn_table(test_display, y_prob, top_n=top_n_customers)

    # Attach complaint text snippet if available
    if has_text and "complaint_text" in test_display.columns:
        text_snippet = (
            test_display["complaint_text"]
            .fillna("")
            .str.slice(0, 60)
            .str.strip()
            .reset_index(drop=True)
        )
        risk_table = risk_table.copy()
        if len(text_snippet) == len(test_display):
            risk_table["complaint_preview"] = (
                test_display["complaint_text"]
                .fillna("")
                .iloc[:len(risk_table)]
                .str.slice(0, 60)
                .values
            )

    def highlight_risk(row):
        colors = {
            "Critical": "background-color: #FEE2E2",
            "High":     "background-color: #FEF3C7",
            "Medium":   "background-color: #DBEAFE",
            "Low":      "",
        }
        return [colors.get(str(row.get("risk_tier", "")), "")] * len(row)

    st.dataframe(
        risk_table.style.apply(highlight_risk, axis=1),
        use_container_width=True,
        height=440,
    )

    csv_buf = io.StringIO()
    risk_table.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download High-Risk Customer List (CSV)",
        data=csv_buf.getvalue(),
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ── Dataset Preview ───────────────────────────────────────────────────────
    with st.expander("Filtered Dataset Preview"):
        st.dataframe(df_filtered.head(100), use_container_width=True)

    st.caption(
        "Telecom Churn Intelligence v2.0 — "
        "Customer IDs are anonymized. No personally identifiable information is stored or transmitted."
    )


if __name__ == "__main__":
    main()
