# Customer Intelligence Platform

A backend platform for customer intelligence — churn prediction, customer segmentation, and targeted retention marketing — built on the Olist Brazilian e-commerce dataset.

This repository implements the end-to-end data pipeline and machine learning layer: CSV ingestion → cleaned MySQL tables → engineered behavioral features → churn labels → encoded & scaled model-ready splits → class imbalance handling → LightGBM and Logistic Regression → calibration → SHAP explainability → customer-level churn predictions → cohort and trend analytics → forecast generation → risk-tier classification + GMM segmentation → CLV scoring → campaign recommendations.

---

## Project Overview

Using the Olist E-Commerce dataset, the platform:

- Cleans and loads 8 raw CSVs into relational MySQL tables.
- Engineers behavioral features spanning payment, review, product, freight, delivery, category, and geographic signals. The final model feature set is selected from the training split and is recorded with each model artifact.
- Derives customer churn labels using a configurable return-window rule (`label_config.yaml`).
- Encodes and scales features into time-split `model_ready_train`, `model_ready_val`, and `model_ready_test` tables.
- Benchmarks 8 class-imbalance strategies (cost-sensitive weighting and resampling) for a 30.26:1 churned-to-retained class ratio.
- Trains LightGBM and Logistic Regression churn classifiers, with threshold analysis, calibration, and odds-ratio interpretability for the logistic baseline.
- Tracks data drift across temporal splits using Population Stability Index (PSI).
- Automatically logs all training runs, thresholds, and confusion matrix metrics to centralized experiment trackers.
- Runs customer cohort analysis, historical trend analysis, and time-series forecasting for churn and revenue behavior.
- Builds a customer intelligence layer with risk-tier classification, Gaussian Mixture Model segmentation, and customer-level CLV scoring.
- Generates campaign recommendation tables using combined risk, segment, and CLV signals for targeted retention actions.

---

## Model Performance Benchmarks

The following held-out test metrics come from the current model comparison. Both
models use **churn (`churn_label = 1`) as the positive class**, and their
operating thresholds are tuned on the validation split before final evaluation.

| Model | ROC-AUC | PR-AUC (Avg Prec) | Precision | Recall | Balanced Acc | Primary Use Case |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LightGBM** | **`0.9492`** | **`0.9956`** | **`99.12%`** | **`88.64%`** | **`87.92%`** | Automated churn-risk targeting |
| **Logistic Regression** | **`0.8770`** | **`0.9892`** | **`99.19%`** | **`70.53%`** | **`80.61%`** | Auditable strategic insights |

The LightGBM training artifact also reports a top-5% *retention* segment, where
the positive class is retained customers (`churn_label = 0`). Do not compare
those retention-segment metrics directly with the churn metrics above.

---

## Current Model Feature Contract (13 Features)

The current deployed LightGBM artifact expects the 13 model-ready features
below. Their values must use the same encoding and scaling applied during
training. Feature selection can change this contract after retraining, so
update this list whenever a new artifact is deployed.

| # | Feature Name | Type | Description / Business Intuition |
| :-: | :--- | :---: | :--- |
| 1 | `monetary_value` | Continuous | Total customer spend in Brazilian Real (BRL) |
| 2 | `avg_payment_installments` | Continuous | Average installment split per transaction (credit risk & purchasing behavior) |
| 3 | `avg_review_score` | Continuous | Customer satisfaction rating (1.0 to 5.0 scale) |
| 4 | `has_bad_review` | Binary (0/1) | Flag indicating customer left a 1-star or 2-star review |
| 5 | `has_review_comment` | Binary (0/1) | Flag indicating customer wrote explicit text in their review (high-engagement / vocal customer) |
| 6 | `avg_product_weight_g` | Continuous | Physical weight footprint of ordered items in grams (correlates with bulky logistics) |
| 7 | `freight_ratio` | Continuous | Freight value divided by total payment (shipping cost burden relative to cart size) |
| 8 | `avg_delivery_days` | Continuous | Transit duration from purchase date to carrier delivery date |
| 9 | `avg_delivery_delay_days` | Continuous | Elapsed days between carrier delivery date and carrier estimated delivery date |
| 10 | `is_delayed_delivery` | Binary (0/1) | Flag indicating delivery arrived past the estimated delivery date |
| 11 | `dominant_product_category_frequency` | Continuous | Frequency encoding of the customer's primary product category |
| 12 | `customer_city_state_frequency` | Continuous | Frequency encoding of customer geographic location (captures logistics density) |
| 13 | `preferred_payment_type_debit_card` | Binary (0/1) | One-hot encoded payment preference: Debit Card |

---

## Class Imbalance Handling & Strategy Benchmarks

### 1. Imbalance Profile (Training Split)
- **Total Training Samples**: 48,232 customers
- **Retained Customers (Class 0 / Minority)**: 1,543 (3.20%)
- **Churned Customers (Class 1 / Majority)**: 46,689 (96.80%)
- **Class Ratio**: **30.26 : 1** churned-to-retained
- **Theoretical 'Balanced' Weights**: Class 0 = `15.6293x`, Class 1 = `0.5165x` (effective penalty ratio: **30.26x** on minority errors).
- **Core Implication**: A naive classifier achieves 96.8% accuracy simply by predicting every customer churns, resulting in **0.0% recall** on retained customers. Cost-sensitive loss weighting or threshold tuning is mandatory.

### 2. Strategy Benchmarks — Logistic Regression
Evaluated using validation-tuned decision thresholds on the validation set:

| Strategy | Tuned Thresh | Balanced Acc | Retained Recall | Retained Prec | Churn Recall | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cost-Sensitive (15:1)** | **0.71** | **0.8088** | **88.5%** | 17.0% | 73.2% | 0.8758 | 0.9890 |
| **Cost-Sensitive (45:1)** | 0.49 | 0.8086 | 90.2% | 16.4% | 71.5% | 0.8772 | 0.9892 |
| **Inverse Frequency ('balanced')** | 0.59 | 0.8061 | 90.7% | 16.0% | 70.5% | 0.8770 | 0.9892 |
| **Undersampling (1:1)** | 0.59 | 0.8065 | 90.7% | 16.0% | 70.6% | 0.8771 | 0.9892 |
| **Undersampling (3:1)** | 0.78 | 0.8038 | 87.7% | 16.8% | 73.0% | 0.8729 | 0.9888 |
| **Oversampling (1:1)** | 0.59 | 0.8043 | 90.5% | 15.9% | 70.3% | 0.8765 | 0.9891 |
| **Cost-Sensitive (5:1)** | 0.87 | 0.7967 | 86.1% | 16.6% | 73.3% | 0.8706 | 0.9884 |
| **Unweighted Baseline** | 0.97 | 0.7800 | 84.2% | 15.6% | 71.8% | 0.8575 | 0.9869 |

### 3. Strategy Benchmarks — LightGBM
Evaluated across cost-sensitive weights and resampling techniques:

| Strategy | Tuned Thresh | Balanced Acc | Retained Recall | Retained Prec | Churn Recall | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unweighted Baseline + Thresh Tuning** | **0.97** | **0.8822** | **88.0%** | **31.9%** | **88.4%** | **0.9483** | **0.9956** |
| **Cost-Sensitive (5:1)** | 0.94 | 0.8691 | 92.0% | 23.8% | 81.8% | 0.9494 | 0.9959 |
| **Cost-Sensitive (15:1)** | 0.87 | 0.8691 | 91.9% | 23.9% | 82.0% | 0.9466 | 0.9956 |
| **Inverse Frequency ('balanced')** | 0.78 | 0.8690 | 91.0% | 24.6% | 82.8% | 0.9445 | 0.9955 |
| **Cost-Sensitive (45:1)** | 0.77 | 0.8661 | 92.7% | 22.8% | 80.5% | 0.9431 | 0.9953 |
| **Undersampling (3:1)** | 0.81 | 0.8669 | 88.7% | 26.4% | 84.7% | 0.9421 | 0.9955 |
| **Undersampling (1:1)** | 0.61 | 0.8593 | 90.7% | 22.9% | 81.2% | 0.9373 | 0.9951 |
| **Oversampling (1:1)** | 0.85 | 0.8568 | 93.5% | 20.7% | 77.8% | 0.9442 | 0.9954 |

### 4. Key Takeaways
1. **Select a strategy by the target operating metric.** In this run, the unweighted LightGBM baseline with threshold tuning produced the strongest balanced accuracy (`0.8822`), while other strategies trade retained-customer recall, churn recall, and precision differently.
2. **Threshold tuning is part of model selection.** Thresholds are chosen on the validation split; the test set is reserved for final reporting.
3. **Keep the target class explicit.** The benchmark tables report retained-customer and churn recall separately, and PR-AUC is calculated for the stated positive class.

---

## Customer Segmentation & Campaign Intelligence

Once a customer has a churn probability (from `churn_predictions`), the
segmentation stack turns that single number — plus behavioral features and
spend history — into three business-facing signals, then combines them into
a recommended retention action.

### Risk Tier Classification (schema 6.1)

Each customer's `churn_probability` is mapped to one of three tiers using
cutoffs defined in `app/segmentation/customer_intelligence/config/risk_tier_thresholds.yaml`
(editable without a code change):

| Risk Tier | Churn Probability Range |
| :--- | :---: |
| Low Risk | 0.00 – 0.39 |
| Medium Risk | 0.40 – 0.69 |
| High Risk | 0.70 – 1.00 |

### GMM Customer Segmentation (schema 6.2)

A Gaussian Mixture Model clusters customers on 11 behavioral features
(`monetary_value`, `frequency`, `recency_days`, `avg_review_score`,
`avg_delivery_days`, `avg_delivery_delay_days`, `avg_payment_installments`,
`freight_ratio`, `has_bad_review`, `has_review_comment`, `is_delayed_delivery`)
— identifiers and churn probability are deliberately excluded from the
clustering itself. The number of clusters (k) is chosen by evaluating
`k = 2..7` against BIC, Silhouette, and Davies-Bouldin scores together rather
than BIC alone, plus a practical minimum-cluster-size floor. GMM only ever
outputs a numeric `segment_id` and a `cluster_probability` confidence score;
business-friendly `segment_label`s (e.g. a value/experience/loyalty
combination) are assigned afterward by profiling each cluster's average
characteristics — segment labels never encode risk, since that's already
captured separately by the risk tier.

### Customer Lifetime Value (CLV)

CLV is derived entirely from features Feature Engineering has already
computed — no re-query of raw transactions:

```text
clv = avg_order_value × purchase_frequency_per_year × customer_lifespan_years
```

`purchase_frequency_per_year` falls back to the cohort's average annual
frequency for one-time buyers (who have no individual rate to observe).
Customers are then split into `value_tier` — High / Medium / Low — using a
percentile split by default (top third / middle third / bottom third of CLV),
which can be overridden with fixed business cutoffs via
`ValueTierThresholds` in `clv_calculator.py`.

### Campaign Recommendation Engine (schema 6.3)

The final stage looks up each customer's `(risk_tier, segment, value_tier)`
combination against `app/segmentation/customer_intelligence/campaign_rules_reasoncodes.yaml`
and produces a `campaign_name`, `campaign_priority`, and `reason_code` — for
example, high-risk + high-value customers route to high-touch retention
outreach, while low-value segments get lighter-weight engagement rather than
blanket discounts. Rules are edited in YAML, not code, and are validated at
load time by `campaign_rules_loader.py`.

Run the whole stack, or any stage of it individually, with `scripts/run_pipeline.py`
(see [Running the Pipeline](#running-the-pipeline)) or via the standalone
`-m` commands in [§6](#6-run-the-segmentation-stack-directly).

---

## Tech Stack

| Component | Tool / Library | Usage |
| :--- | :--- | :--- |
| **Runtime** | Python 3.11+ | Execution environment |
| **Data & Feature Engineering** | Pandas, NumPy, scikit-learn | Cleaning, aggregation, frequency encoding, RobustScaler |
| **Database** | MySQL 8.x, SQLAlchemy, PyMySQL | Relational storage for raw, feature, and model-ready tables |
| **Machine Learning** | LightGBM, scikit-learn | Gradient boosted trees and cost-sensitive logistic regression |
| **Hyperparameter Tuning** | Optuna (TPE Sampler) | Optional 5-fold stratified optimization on PR-AUC |
| **Data Drift & Monitoring** | Population Stability Index (PSI) | Tracking feature & score stability across Train/Val/Test |
| **Model Persistence** | Joblib | Production bundle packaging (`.joblib` & `.json` metadata) |
| **Explainability & Serving** | SHAP, FastAPI, Uvicorn, Pydantic | Feature contributions, request/response validation, and HTTP prediction API |
| **Time Series Forecasting** | statsmodels (ETS / Exponential Smoothing) | Revenue and churn trend forecasting |
| **Customer Segmentation** | scikit-learn (Gaussian Mixture Models), PyYAML | GMM clustering, risk-tier thresholds, and campaign rules config |
| **Testing** | Pytest | Automated unit and integration test suite |

---

## Team

| Module | Task | Owner |
| :--- | :--- | :--- |
| Data Engineering | Ingestion, Cleaning & MySQL Schema | Yashaswi |
| Feature Engineering | RFM, Product, Review & Fulfillment Features | Lohit |
| Machine Learning | Logistic Regression Modeling & Odds Ratios | Kalyan |
| Segmentation & Campaigns | Customer Clustering & Marketing Slices | Kuushalie |
| FastAPI & Integration | Serving APIs & Dashboard Integration | Rajeswari |

---

## Repository Structure

```text
customer-intelligence-platform/
├── app/
│   ├── Database/
│   │   ├── cleaning.py                   # Data cleaning rules and null handling
│   │   ├── database.py                   # SQLAlchemy MySQL engine connection
│   │   └── ingest.py                     # CSV to MySQL ingestion pipeline
│   ├── Features/
│   │   ├── build_features.py             # Feature engineering (freight, weight, spend, reviews)
│   │   ├── build_churn_label.py          # Churn label derivation from order timestamps
│   │   ├── merge.py                      # Merges customer features with churn labels
│   │   ├── encoding_transformation.py    # Frequency and categorical encoding
│   │   └── feature_selection_and_scaling.py # Feature selection & RobustScaler splits
│   ├── ml/
│   │   ├── logistic_regression.py        # Cost-sensitive Logistic Regression & odds ratios
│   │   ├── train_lightgbm_model.py       # Production LightGBM model, PSI drift & Optuna tuning
│   │   ├── imbalance_experiments.py      # 8-strategy class imbalance benchmark suite
│   │   ├── metrics.py                    # Evaluation metrics & threshold search functions
│   │   ├── calibration.py                 # Probability calibration and calibrated artifacts
│   │   ├── threshold_analysis.py          # Validation threshold sweeps
│   │   ├── model_comparison.py            # Logistic Regression vs LightGBM comparison
│   │   ├── run_all.py                     # ML-only orchestration
│   │   ├── run_calibration.py             # Calibration entry point
│   │   ├── experiment_logger.py            # Centralized CSV/Markdown experiment tracker
│   │   └── explainibility_inference/      # API, inference, SHAP, reason codes, batch scoring
│   ├── config/
│   │   ├── label_config.yaml             # Configurable churn observation & return windows
│   │   └── trend_config.yaml             # Historical-trend configuration
│   └── segmentation/
│       ├── customer_analytics/           # Cohort, trend, and forecasting pipelines
│       │   ├── cohort_analysis.py         # Retention, repeat-purchase, revenue & churn by cohort
│       │   ├── Trends/                    # Historical trend loading, computation & export
│       │   │   ├── data_loader.py
│       │   │   ├── trend_engine.py
│       │   │   ├── export.py
│       │   │   ├── schemas.py
│       │   │   └── pipeline.py            # Orchestrates the historical trend stage
│       │   └── forecasting/
│       │       └── forecasting.py         # ETS-based revenue & churn trend forecasting
│       └── customer_intelligence/        # Risk tiers, GMM segmentation, CLV, campaign recommendations
│           ├── data_access/
│           │   ├── customer_intelligence_repository.py  # Merges features_encoded + churn_predictions
│           │   └── campaign_input_repository.py          # Merges risk_tier + segment + CLV value_tier
│           ├── risk_tier/
│           │   └── classifier.py          # Schema 6.1 — churn_probability -> risk_tier
│           ├── gmm/
│           │   ├── feature_prep.py        # Selects behavioural features for clustering
│           │   ├── model_selection.py     # BIC + Silhouette + Davies-Bouldin cluster-count selection
│           │   ├── segmenter.py           # Trains final GMM, assigns segment_id & cluster_probability
│           │   └── labeling.py            # Assigns business labels to trained clusters
│           ├── pipeline/
│           │   └── generate_customer_intelligence_tables.py  # Orchestrates risk tier + GMM segmentation
│           ├── campaign_engine/
│           │   ├── campaign_rules_loader.py   # Loads & validates campaign_rules_reasoncodes.yaml
│           │   ├── recommender.py             # Schema 6.3 — Campaign Recommendation Engine
│           │   └── pipeline/
│           │       └── generate_campaign_recommendations_table.py
│           ├── config/
│           │   └── risk_tier_thresholds.yaml  # Configurable risk-tier cutoffs
│           ├── campaign_rules_reasoncodes.yaml  # Campaign rules & reason codes (schema 6.3)
│           ├── clv_calculator.py           # Customer lifetime value calculation
│           └── generate_clv_table.py       # Persists CLV output to customer_clv table
├── data/
│   ├── Schema.sql                        # DDL schema with foreign keys and indexes
│   └── olist_*.csv                       # Source raw CSVs
├── outputs/                              # Local model artifacts & reports (git-ignored)
│   ├── models/                           # Serialized .joblib bundles, metadata .json, and plots
│   └── reports/                          # experiment_log.csv, imbalance benchmarks
├── scripts/
│   └── run_pipeline.py                   # Orchestrates end-to-end data & modeling pipeline
├── tests/                                # Automated tests for database, features, and ML models
│   ├── test_build_churn_label.py
│   ├── test_build_features.py
│   ├── test_cleaning.py
│   ├── test_encoding_transformation.py
│   ├── test_experiment_logger.py
│   ├── test_feature_selection_and_scaling.py
│   ├── test_imbalance_experiments.py
│   ├── test_lightgbm_model.py            # 16 LightGBM contracts & PSI drift tests
│   ├── test_logistic_regression.py
│   └── test_merge.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone <repository-url>
cd customer-intelligence-platform
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

### 3. Install Dependencies
```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

### 4. Database Setup
Create the MySQL database and load the schema:
```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS olist CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
mysql -u root -p olist < data/Schema.sql
```

### 5. Environment Variables
Copy `.env.example` to `.env` and configure your database credentials:
```bash
cp .env.example .env
```
Ensure `DB_USER`, `DB_PASSWORD`, `DB_NAME`, and `DATASET_DIR` are populated.
If you will generate predictions, configure the model path and version before
running the complete pipeline. Calibration writes the artifact to this path:

```env
MODEL_PATH=outputs/models/lgb_churn_model_calibrated.joblib
MODEL_VERSION=v1
```

`MODEL_PATH` may be relative to the repository root. Use the exact artifact
created by calibration if its filename differs. `MODEL_VERSION` is stored with
each prediction record.

---

## Running the Pipeline

The pipeline script [`scripts/run_pipeline.py`](./scripts/run_pipeline.py) orchestrates the entire workflow:

### 1. Build the database, feature, and analytics tables

By default, the script also runs prediction generation, cohort and trend
analytics, forecasting, segmentation, CLV, and campaign recommendations.
Those downstream stages require their upstream tables to exist. To build only
the database, feature, and analytics inputs without refreshing predictions or
running downstream customer-intelligence stages:

```bash
# Ingest the CSV files and build the prerequisite tables:
.venv/bin/python scripts/run_pipeline.py \
  --skip-predictions \
  --skip-segmentation \
  --skip-clv \
  --skip-campaign

# Skip ingestion when raw tables already exist:
.venv/bin/python scripts/run_pipeline.py \
  --skip-ingest \
  --skip-predictions \
  --skip-segmentation \
  --skip-clv \
  --skip-campaign
```

### 2. Run the complete pipeline

This command builds the feature tables, runs the churn ML workflow, refreshes
customer predictions, performs cohort and historical trend analysis,
constructs forecast tables, and runs the segmentation stack (risk tiers,
GMM customer segments, CLV, and campaign recommendations):

```bash
.venv/bin/python scripts/run_pipeline.py --skip-ingest --train-all
```

By default, the pipeline runs ingestion, feature preparation, prediction
generation, analytics, and the downstream customer-intelligence modules. It
does not train models unless one of the training flags is provided. Use
`--train-all` for the complete ML workflow, or use
`--skip-predictions`, `--skip-segmentation`, `--skip-clv`, or
`--skip-campaign` when those stages are already available or intentionally not
needed. Individual stages can be skipped when their inputs or outputs are
already available:

```bash
.venv/bin/python scripts/run_pipeline.py --skip-cohort
.venv/bin/python scripts/run_pipeline.py --skip-trends
.venv/bin/python scripts/run_pipeline.py --skip-forecasting
.venv/bin/python scripts/run_pipeline.py --skip-segmentation
.venv/bin/python scripts/run_pipeline.py --skip-clv
.venv/bin/python scripts/run_pipeline.py --skip-campaign
.venv/bin/python scripts/run_pipeline.py --skip-predictions
```

The full pipeline creates or refreshes these tables:

- `model_ready_train`
- `model_ready_val`
- `model_ready_test`
- `churn_predictions`
- `customer_cohort_analysis`
- `historical_revenue_trend`
- `historical_churn_trend`
- `historical_trend_combined`
- `forecast_revenue_trend`
- `forecast_churn_trend`
- `forecast_trend_combined`
- `customer_intelligence_base`
- `customer_risk_tiers`
- `customer_segments`
- `customer_clv`
- `customer_campaign_recommendations`

### 3. Run only the ML pipeline

Use this when `model_ready_train`, `model_ready_val`, and
`model_ready_test` already exist:

```bash
.venv/bin/python -m app.ml.run_all

# Skip the optional class-imbalance benchmark:
.venv/bin/python -m app.ml.run_all --skip-imbalance

# Re-run analysis, comparison, and calibration using existing model artifacts:
.venv/bin/python -m app.ml.run_all --skip-training
```

The ML-only pipeline includes:

- Logistic Regression training
- LightGBM training
- Class-imbalance experiments
- Threshold analysis
- Model comparison
- LightGBM calibration

It does not run cohort analysis, historical trends, forecasting, or batch
prediction generation. Use `scripts/run_pipeline.py` for the complete workflow.

### 4. Run individual ML modules

```bash
.venv/bin/python -m app.ml.logistic_regression
.venv/bin/python -m app.ml.train_lightgbm_model
.venv/bin/python -m app.ml.imbalance_experiments
.venv/bin/python -m app.ml.threshold_analysis
.venv/bin/python -m app.ml.model_comparison
```

### 5. Run the explainability interface

The explainability interface loads the configured calibrated model, validates
the model feature contract, generates churn predictions, calculates SHAP
feature contributions, and returns business reason codes.

Run the local smoke check without starting the API:

```bash
.venv/bin/python -m app.ml.explainibility_inference.smoke_check
```

Start the FastAPI service:

```bash
.venv/bin/python -m uvicorn \
  app.ml.explainibility_inference.api.main:app \
  --reload \
  --host 127.0.0.1 \
  --port 8000
```

Open the interactive API documentation at
[`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs).

Available endpoints:

```text
GET  /health
POST /api/v1/predictions/from-features
GET  /api/v1/predictions/{customer_unique_id}
POST /api/v1/predictions/batch
```

Use `/predictions/from-features` when the caller already has all 13 required
model-ready feature values. The API does not transform raw customer data, so
the supplied values must use the same encoding and scaling as the deployed
model. The request body has this shape:

```json
{
  "customers": [
    {
      "customer_unique_id": "C001",
      "features": {
        "monetary_value": 1.2,
        "avg_payment_installments": 0.0,
        "avg_review_score": 0.0,
        "has_bad_review": 0,
        "has_review_comment": 1,
        "avg_product_weight_g": 0.4,
        "freight_ratio": 0.1,
        "avg_delivery_days": 0.2,
        "avg_delivery_delay_days": 0.0,
        "is_delayed_delivery": 0,
        "dominant_product_category_frequency": 0.03,
        "customer_city_state_frequency": 0.01,
        "preferred_payment_type_debit_card": 1
      }
    }
  ]
}
```

The API reads `MODEL_PATH` and `MODEL_VERSION` from `.env`. Run commands from
the repository root. Using `.venv/bin/python` ensures that Uvicorn, SHAP, and
the model use the same environment.

The prediction response includes:

- `churn_probability`
- `shap_values`
- `reason_codes`
- `model_version`
- `scored_at`

The full-customer prediction job reads `features_encoded`, scores eligible
customers, and refreshes the `churn_predictions` table:

```bash
.venv/bin/python -m app.ml.explainibility_inference.inference.generate_predictions_table
```

Run this command after calibration and after confirming that `MODEL_PATH` and
`MODEL_VERSION` are set in `.env`:

```bash
# .env
MODEL_PATH=outputs/models/lgb_churn_model_calibrated.joblib
MODEL_VERSION=v1

# from the repository root
.venv/bin/python -m app.ml.explainibility_inference.inference.generate_predictions_table
```

This job reads every eligible customer from `features_encoded`, generates
churn probabilities, SHAP values, and reason codes, and replaces the
`churn_predictions` table.

The prediction-table job is also run automatically by `scripts/run_pipeline.py`
after model training and calibration. Run the module separately only when a
standalone prediction refresh is needed.

### 6. Run the segmentation stack directly

The segmentation stage (risk tiers, GMM segmentation, CLV, and campaign
recommendations) can also be run stage by stage, independent of
`scripts/run_pipeline.py`, once `churn_predictions` and `features_encoded`
already exist. Run these in order — CLV has no upstream dependency here, but
Campaign Recommendations requires both Risk Tier + GMM Segmentation and CLV
to have already written their tables:

```bash
# 1. Risk Tier Classification + GMM Segmentation
#    writes: customer_intelligence_base, customer_risk_tiers, customer_segments
.venv/bin/python -m app.segmentation.customer_intelligence.pipeline.generate_customer_intelligence_tables

# 2. Customer Lifetime Value
#    writes: customer_clv
.venv/bin/python -m app.segmentation.customer_intelligence.generate_clv_table

# 3. Campaign Recommendations (run last — depends on the two tables above)
#    writes: customer_campaign_recommendations
.venv/bin/python -m app.segmentation.customer_intelligence.campaign_engine.pipeline.generate_campaign_recommendations_table
```

Risk-tier cutoffs live in `app/segmentation/customer_intelligence/config/risk_tier_thresholds.yaml`
and campaign rules live in `app/segmentation/customer_intelligence/campaign_rules_reasoncodes.yaml` —
both are editable without touching application code.

---

## Expected Table Outputs in MySQL

A successful pipeline run creates or refreshes the following relational tables.
Row and column counts depend on the source snapshot, cleaning rules, feature
selection, observation window, and model eligibility filters, so the values
below describe expected contents rather than a fixed schema.

| Stage | Table | Description |
| :--- | :--- | :--- |
| Features | `customer_features` | Customer behavioral aggregations |
| Labels | `customer_churn_labels` | Churn flags, dates, recency, order count, and censoring status |
| Merge | `customer_features_with_labels` | Joined features and targets |
| Encode | `features_encoded` | Frequency-encoded and transformed features |
| Split | `model_ready_train` | Temporal training split |
| Split | `model_ready_val` | Validation split for threshold tuning |
| Split | `model_ready_test` | Unseen test split for final reporting |
| Predictions | `churn_predictions` | Eligible-customer probabilities, SHAP values, and reason codes |
| Cohort | `customer_cohort_analysis` | Retention, repeat purchase, revenue, and churn by cohort and relative month |
| Trends | `historical_revenue_trend` | Historical revenue trend by configured granularity |
| Trends | `historical_churn_trend` | Historical churn trend by configured granularity |
| Trends | `historical_trend_combined` | Combined historical revenue and churn trends |
| Forecast | `forecast_revenue_trend` | Forecast revenue values and intervals |
| Forecast | `forecast_churn_trend` | Forecast churn values and intervals |
| Forecast | `forecast_trend_combined` | Combined revenue and churn forecasts |
| Segmentation | `customer_intelligence_base` | merged customer intelligence dataset used for risk & segmentation |
| Segmentation | `customer_risk_tiers` | customer risk assignment derived from churn probability |
| Segmentation | `customer_segments` | GMM segment membership and segment labels |
| CLV | `customer_clv` | customer lifetime value score and output table |
| Campaigns | `customer_campaign_recommendations` | recommended retention actions using segment, risk, and CLV inputs |

---

## Testing

Run the full automated test suite from the repository root:

```bash
pytest
```

The suite contains test modules covering ingestion and cleaning, feature
engineering, database access, cohort analysis, forecasting, historical trends,
class-imbalance handling, Logistic Regression, LightGBM, calibration,
segmentation, CLV, campaign recommendations, and the explainability/prediction
interface. Use focused runs while developing:

```bash
pytest tests/test_cohort_analysis.py
pytest tests/test_trend_analysis.py tests/test_forecasting.py
pytest tests/explainibility_interface_tests/
```

The complete suite requires the dependencies in the project environment,
including the explainability stack. If collection reports a missing optional
dependency such as `shap`, install the project dependencies before interpreting
the result as a test failure.

---

## License

This repository is intended for internship training and internal educational purposes.
