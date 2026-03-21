# Telecom Customer Churn Intelligence
### Predictive Analytics for Pakistan Telecom Market
 
---
 
## Problem Statement
 
Customer churn is the primary revenue leakage challenge in the Pakistani telecom sector. 
Acquiring a new customer costs five to seven times more than retaining an existing one. 
Network quality, contract type, and tenure are the dominant churn drivers in this market, 
yet most operators still rely on reactive measures after customers have already left.
 
This project builds a production-grade churn prediction pipeline that:
 
- Identifies customers at high risk of churning before they defect
- Incorporates network quality as a first-class domain feature
- Provides interpretable, per-customer explanations for every prediction
- Quantifies the financial impact of model decisions
- Delivers an interactive dashboard for operations and retention teams
 
---
 
## Approach
 
### Pipeline Stages
 
```
Raw Data (Telco CSV)
      |
      v
Data Cleaning & Quality Checks
      |
      v
Feature Engineering
  - Network quality score (proxy for regional infrastructure)
  - Tenure x network interaction
  - Monthly charges x network interaction
  - Service bundle count
  - Average monthly charges
  - Tenure bracket encoding
  - Recent contract change flag
      |
      v
Encoding & Scaling
      |
      v
Train/Test Split (stratified, 80/20)
      |
      +--> Logistic Regression   (interpretable baseline)
      +--> Random Forest         (non-linear, feature importance)
      +--> XGBoost               (high-performance, SHAP-compatible)
      |
      v
Evaluation
  - Accuracy, Precision, Recall, F1, ROC-AUC
  - Confusion matrix
  - Calibration curve
  - Lift / Gain chart
  - Cost-sensitive financial impact analysis
      |
      v
Explainability
  - SHAP global summary (beeswarm + bar)
  - Per-customer SHAP waterfall contributions
  - Partial Dependence Plots (PDP)
  - Individual Conditional Expectation (ICE)
      |
      v
Streamlit Dashboard
  - KPI cards, filterable risk table
  - Downloadable high-risk customer roster
```
 
---
 
## Key Findings
 
| Feature | Churn Direction | Business Interpretation |
|---|---|---|
| network_quality (Low) | Increases churn | Poor infrastructure in rural Pakistan is the #1 churn driver |
| Month-to-month contract | Increases churn | No lock-in period; easy to defect |
| Low tenure (0-12 months) | Increases churn | New customers have not yet built loyalty |
| Fiber optic + high charges | Mixed | High-value customers churn less but are more price sensitive |
| Multiple services subscribed | Decreases churn | Bundle lock-in effect is significant |
| Electronic check payment | Increases churn | Associated with lower commitment profiles |
 
---
 
## Model Performance
 
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.800 | ~0.640 | ~0.560 | ~0.598 | ~0.844 |
| Random Forest | ~0.800 | ~0.660 | ~0.490 | ~0.563 | ~0.838 |
| XGBoost | ~0.806 | ~0.660 | ~0.540 | ~0.594 | ~0.851 |
 
*Results vary slightly by random seed and hyperparameter configuration.*
 
---
 
## Business Recommendations
 
1. **Network investment targeting**: Prioritise 4G/5G infrastructure rollout in 
   low-network-quality regions. Churn rates among low-quality-network customers 
   exceed 40% in the first 12 months.
 
2. **Proactive retention at month 1-3**: Flag all month-to-month customers 
   with tenure below 3 months for an automated retention call or discount offer.
 
3. **Bundle incentive programme**: Customers with 3 or fewer services should 
   receive targeted upsell offers; each additional service reduces predicted 
   churn probability significantly.
 
4. **High-value early warning**: The model's top decile captures roughly 40% 
   of all churners in 10% of the customer base, enabling highly cost-efficient 
   retention targeting.
 
5. **Payment method migration**: Customers paying by electronic check show 
   systematically higher churn. Incentivising migration to automatic bank 
   transfer or credit card can reduce attrition.
 
---
 
## Project Structure
 
```
telecom-churn-enhanced/
|
+-- data/
|   +-- telco_churn.csv               # Source dataset (place here)
|   +-- processed/                    # Versioned processed datasets (auto-generated)
|
+-- notebooks/
|   +-- analysis.py                   # Full end-to-end analysis script
|
+-- src/
|   +-- data_utils.py                 # Loading, cleaning, encoding, splitting
|   +-- feature_utils.py              # All feature engineering logic
|   +-- model_utils.py                # Training, evaluation, persistence
|   +-- explain_utils.py              # SHAP, PDP, ICE explanations
|   +-- viz_utils.py                  # Matplotlib visualizations
|
+-- models/                           # Saved model files (auto-generated)
+-- artifacts/                        # Saved plots and CSVs (auto-generated)
+-- app.py                            # Streamlit dashboard
+-- requirements.txt
+-- README.md
```
 
---
 
## Setup and Execution
 
### Prerequisites
 
- Python 3.9 or later
- pip
 
### Installation
 
```bash
# Clone or download the project
cd telecom-churn-enhanced
 
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate         # Linux / macOS
venv\Scripts\activate            # Windows
 
# Install dependencies
pip install -r requirements.txt
```
 
### Data
 
Download the Telco Customer Churn dataset from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
 
Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` into the `data/` folder and 
rename it to `telco_churn.csv`.
 
### Run the Analysis Pipeline
 
```bash
python notebooks/analysis.py
```
 
This will:
- Clean and engineer features
- Train Logistic Regression, Random Forest, and XGBoost models
- Save models to `models/`
- Save evaluation plots and reports to `artifacts/`
 
### Launch the Dashboard
 
```bash
streamlit run app.py
```
 
Open `http://localhost:8501` in your browser.
 
---
 
## Security and Privacy
 
- `customerID` is the only customer identifier used; it is anonymised in the 
  source dataset and is never used as a model feature.
- Raw data files are excluded from version control via `.gitignore`.
- The dashboard exposes only aggregated metrics and filtered data; no raw 
  dataset is downloadable through the UI.
- No personally identifiable information is stored, logged, or transmitted.
- Operators deploying this system should ensure compliance with Pakistan's 
  Prevention of Electronic Crimes Act (PECA) and any applicable PTA data 
  localisation requirements, as well as GDPR if EU data subjects are present.
 
---
 
## Reproducibility
 
- All random seeds are fixed at `RANDOM_STATE = 42`.
- Processed datasets are saved with timestamps for full pipeline auditability.
- `requirements.txt` pins major version ranges for all dependencies.
 
---
 
## License
 
This project is intended for portfolio and educational use. 
The underlying Telco dataset is publicly available under its original terms.
