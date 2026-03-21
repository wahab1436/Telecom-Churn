"""
notebooks/analysis.py
---------------------
End-to-end exploratory analysis and model training script.
Run this to reproduce all results, generate saved models,
and produce evaluation artifacts.

Usage:
    python notebooks/analysis.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Project root on path ────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_utils import (
    load_data, clean_total_charges, remove_duplicates,
    check_data_quality, encode_categorical, split_train_test,
    scale_numeric_features, save_versioned_dataset,
)
from feature_utils import run_feature_engineering
from model_utils import (
    train_logistic_regression, train_random_forest, train_xgboost,
    evaluate_model, get_feature_importance, save_model,
    cost_sensitive_evaluation,
)
from viz_utils import (
    plot_churn_by_feature, plot_numeric_distribution,
    plot_correlation_heatmap, plot_confusion_matrix,
    plot_roc_curve, plot_precision_recall_curve,
    plot_calibration_curve, plot_lift_chart, display_churn_table,
)

# ── Directories ─────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(ROOT, "data", "telco_churn.csv")
ARTIFACT_DIR = os.path.join(ROOT, "artifacts")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def save_fig(fig: plt.Figure, name: str) -> None:
    path = os.path.join(ARTIFACT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# 1. DATA LOADING AND QUALITY CHECK
# ============================================================================

print("\n" + "=" * 60)
print("STEP 1: Data Loading and Quality Check")
print("=" * 60)

df_raw = load_data(DATA_PATH)
print(f"\nShape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")

report = check_data_quality(df_raw)
print(f"\nMissing values:\n{pd.Series(report['missing_counts'])}")

df = clean_total_charges(df_raw.copy())
df = remove_duplicates(df)

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 60)
print("STEP 2: Feature Engineering")
print("=" * 60)

df = run_feature_engineering(df)

print(f"\nEngineered features preview:")
print(df[["tenure", "network_quality", "tenure_network_interaction",
          "charges_network_interaction", "service_bundle_count",
          "avg_monthly_charges", "tenure_bracket", "recent_contract_change"]].head())

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "=" * 60)
print("STEP 3: Exploratory Data Analysis")
print("=" * 60)

churn_rate = df["Churn"].map({"Yes": 1, "No": 0}).mean()
print(f"\nOverall churn rate: {churn_rate:.2%}")

# Save EDA plots
seg_features = ["network_quality", "Contract", "tenure_bracket", "InternetService"]
for feat in seg_features:
    if feat in df.columns:
        try:
            fig = plot_churn_by_feature(df, feat)
            save_fig(fig, f"eda_churn_by_{feat}.png")
        except Exception as e:
            print(f"  Warning: Could not plot {feat}: {e}")

for num_feat in ["tenure", "MonthlyCharges", "TotalCharges"]:
    try:
        fig = plot_numeric_distribution(df, num_feat)
        save_fig(fig, f"eda_dist_{num_feat}.png")
    except Exception as e:
        print(f"  Warning: {e}")

# ============================================================================
# 4. ENCODING AND SPLITTING
# ============================================================================

print("\n" + "=" * 60)
print("STEP 4: Encoding and Train/Test Split")
print("=" * 60)

df_enc = encode_categorical(df)

save_versioned_dataset(df_enc, os.path.join(ROOT, "data", "processed"))

X_train, X_test, y_train, y_test = split_train_test(df_enc)
X_train_sc, X_test_sc, scaler = scale_numeric_features(X_train.copy(), X_test.copy())

print(f"\nFeature count: {X_train.shape[1]}")
print(f"Train churn rate: {y_train.mean():.2%}")
print(f"Test churn rate:  {y_test.mean():.2%}")

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================

print("\n" + "=" * 60)
print("STEP 5: Model Training")
print("=" * 60)

# Logistic Regression
print("\n[1/3] Logistic Regression...")
lr = train_logistic_regression(X_train_sc, y_train, tune=False)
save_model(lr, os.path.join(MODEL_DIR, "logistic_regression.pkl"))

# Random Forest
print("\n[2/3] Random Forest...")
rf = train_random_forest(X_train, y_train, tune=False)
save_model(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))

# XGBoost
print("\n[3/3] XGBoost...")
xgb_model = train_xgboost(X_train, y_train, tune=False)
if xgb_model:
    save_model(xgb_model, os.path.join(MODEL_DIR, "xgboost.pkl"))

# ============================================================================
# 6. EVALUATION
# ============================================================================

print("\n" + "=" * 60)
print("STEP 6: Model Evaluation")
print("=" * 60)

eval_lr = evaluate_model(lr, X_test_sc, y_test, model_name="Logistic Regression")
eval_rf = evaluate_model(rf, X_test, y_test, model_name="Random Forest")

evals = {
    "Logistic Regression": eval_lr,
    "Random Forest": eval_rf,
}
if xgb_model:
    eval_xgb = evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost")
    evals["XGBoost"] = eval_xgb

# Confusion matrices
for name, ev in evals.items():
    fig_cm = plot_confusion_matrix(y_test, ev["y_pred"])
    save_fig(fig_cm, f"eval_confusion_{name.lower().replace(' ', '_')}.png")

# ROC curves
roc_data = {
    name: {"y_true": y_test.values, "y_prob": ev["y_prob"]}
    for name, ev in evals.items()
}
fig_roc = plot_roc_curve(roc_data)
save_fig(fig_roc, "eval_roc_comparison.png")

# Best model — PR, calibration, lift
best_name = max(evals, key=lambda k: evals[k]["roc_auc"])
best_eval = evals[best_name]
print(f"\nBest model by ROC-AUC: {best_name} ({best_eval['roc_auc']:.4f})")

fig_pr = plot_precision_recall_curve(y_test, best_eval["y_prob"], model_name=best_name)
save_fig(fig_pr, "eval_precision_recall.png")

fig_cal = plot_calibration_curve(y_test, best_eval["y_prob"], model_name=best_name)
save_fig(fig_cal, "eval_calibration.png")

fig_lift = plot_lift_chart(y_test, best_eval["y_prob"])
save_fig(fig_lift, "eval_lift_chart.png")

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 60)
print("STEP 7: Feature Importance")
print("=" * 60)

fi_df = get_feature_importance(rf, list(X_test.columns), top_n=20)
print(f"\nTop 10 features (Random Forest):\n{fi_df.head(10).to_string(index=False)}")

fig_fi, ax = plt.subplots(figsize=(9, 6))
ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="#1D4ED8", alpha=0.85)
ax.set_xlabel("Importance", fontsize=11)
ax.set_title("Top 20 Features — Random Forest", fontsize=12, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig(fig_fi, "feat_importance_rf.png")

# ============================================================================
# 8. COST-SENSITIVE EVALUATION
# ============================================================================

print("\n" + "=" * 60)
print("STEP 8: Cost-Sensitive Evaluation")
print("=" * 60)

cost_result = cost_sensitive_evaluation(
    y_test, best_eval["y_prob"],
    cost_fn=500, cost_fp=50,
)

# ============================================================================
# 9. HIGH-RISK CUSTOMER ROSTER
# ============================================================================

print("\n" + "=" * 60)
print("STEP 9: High-Risk Customer Roster")
print("=" * 60)

df_display = df.copy()
test_display = df_display.loc[y_test.index].reset_index(drop=True)

# Use best model's probabilities
best_model = rf if best_name == "Random Forest" else (xgb_model if xgb_model and best_name == "XGBoost" else lr)
y_prob_best = best_eval["y_prob"]

risk_table = display_churn_table(test_display, y_prob_best, top_n=50)
risk_path = os.path.join(ARTIFACT_DIR, "high_risk_customers.csv")
risk_table.to_csv(risk_path, index=False)
print(f"\nHigh-risk customer roster saved to: {risk_path}")
print(risk_table.head(10).to_string(index=False))

# ============================================================================
# 10. SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

summary_rows = []
for name, ev in evals.items():
    summary_rows.append({
        "Model": name,
        "Accuracy": f"{ev['accuracy']:.4f}",
        "Precision": f"{ev['precision']:.4f}",
        "Recall": f"{ev['recall']:.4f}",
        "F1": f"{ev['f1']:.4f}",
        "ROC-AUC": f"{ev['roc_auc']:.4f}",
    })

summary_df = pd.DataFrame(summary_rows)
print(f"\n{summary_df.to_string(index=False)}")

summary_path = os.path.join(ARTIFACT_DIR, "model_comparison.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nModel comparison saved to: {summary_path}")
print(f"\nAll artifacts saved to: {ARTIFACT_DIR}")
print("\nAnalysis complete.")
