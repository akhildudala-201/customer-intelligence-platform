# Customer Intelligence Platform

A production backend platform for customer intelligence — churn prediction, customer segmentation, and targeted retention marketing — built on the Olist Brazilian e-commerce dataset.

This repository implements the end-to-end data pipeline and machine learning layer: CSV ingestion → cleaned MySQL tables → engineered behavioral features → churn labels → encoded & scaled model-ready splits → class imbalance handling → LightGBM and Logistic Regression → calibration → SHAP explainability → customer-level predictions for segmentation.

---

## Project Overview

Using the Olist E-Commerce dataset, the platform:

- Cleans and loads 8 raw CSVs into relational MySQL tables.
- Engineers 15 non-leaking, high-impact behavioral features (checkout freight burden, product weight, RFM monetary spend, review engagement, logistics delay, category frequencies, and payment preferences).
- Derives customer churn labels using a configurable return-window rule (`label_config.yaml`).
- Encodes and scales features into time-split `model_ready_train`, `model_ready_val`, and `model_ready_test` tables.
- Benchmarks 8 class imbalance strategies (cost-sensitive loss weighting vs. resampling) to handle the 30.26:1 churn skew.
- Trains production-grade churn classifiers:
  - **LightGBM**: Bayesian hyperparameter-tuned tree ensemble achieving **`0.9492 ROC-AUC`** and **`13.2x Lift`** on top 5% risk slice.
  - **Logistic Regression**: High-precision baseline achieving **`0.8770 ROC-AUC`** and **`80.61% Balanced Accuracy`** with odds-ratio interpretability.
- Tracks data drift across temporal splits using Population Stability Index (PSI).
- Automatically logs all training runs, thresholds, and confusion matrix metrics to centralized experiment trackers.

---

## Model Performance Benchmarks

Evaluated on the unseen test set (10,337 customers, 5.82% minority base rate):

| Model | ROC-AUC | PR-AUC (Avg Prec) | Operating Threshold | Precision | Recall | Balanced Acc | Lift over Random | Primary Use Case |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LightGBM** | **`0.9492`** | **`0.7661`** | `Rate Mode (Top 5%)` | **`74.85%`** | **`64.29%`** | — | **`12.85x`** | **Automated Retention Targeting** |
| **Logistic Regression** | **`0.8770`** | **`0.9892`** | `0.59 (Val-tuned)` | **`99.19%`** | **`70.53%`** | **`80.61%`** | — | **Auditable Strategic Insights** |

---

## Engineered Behavioral Features (15 Features)

All features are engineered strictly from the observation window before the temporal split to eliminate target leakage:

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
| 13 | `preferred_payment_type_boleto` | Binary (0/1) | One-hot encoded payment preference: Boleto |
| 14 | `preferred_payment_type_debit_card` | Binary (0/1) | One-hot encoded payment preference: Debit Card |
| 15 | `preferred_payment_type_voucher` | Binary (0/1) | One-hot encoded payment preference: Voucher |

---

## Class Imbalance Handling & Strategy Benchmarks

### 1. Imbalance Profile (Training Split)
- **Total Training Samples**: 48,232 customers
- **Retained Customers (Class 0 / Minority)**: 1,543 (3.20%)
- **Churned Customers (Class 1 / Majority)**: 46,689 (96.80%)
- **Imbalance Ratio**: **30.26 : 1**
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
1. **Cost-Sensitive Loss Weighting > Resampling**:
   - Random undersampling discards up to **95% of the majority class data**, severely destroying sample variance and boundary fidelity.
   - Oversampling duplicates minority records, causing tree ensembles to memorize specific repeated samples.
   - Cost-sensitive loss weighting retains 100% of data while penalizing minority class misclassifications appropriately.
2. **Threshold Tuning is Crucial**: Post-hoc probability threshold search on validation PR curves yields dramatic improvements in balanced accuracy and minority recall without altering the underlying probability geometry.

---

## Tech Stack

| Component | Tool / Library | Usage |
| :--- | :--- | :--- |
| **Runtime** | Python 3.11+ | Execution environment |
| **Data & Feature Engineering** | Pandas, NumPy, scikit-learn | Cleaning, aggregation, frequency encoding, RobustScaler |
| **Database** | MySQL 8.x, SQLAlchemy, PyMySQL | Relational storage for raw, feature, and model-ready tables |
| **Machine Learning** | LightGBM, scikit-learn | Gradient boosted trees and cost-sensitive logistic regression |
| **Hyperparameter Tuning** | Optuna (TPE Sampler) | 5-Fold Stratified Bayesian optimization on PR-AUC |
| **Data Drift & Monitoring** | Population Stability Index (PSI) | Tracking feature & score stability across Train/Val/Test |
| **Model Persistence** | Joblib | Production bundle packaging (`.joblib` & `.json` metadata) |
| **Explainability & Serving** | SHAP, FastAPI, Uvicorn | Feature contributions and HTTP prediction API |
| **Testing** | Pytest | 169 automated tests |

---

## Team

| Module | Task | Owner |
| :--- | :--- | :--- |
| Data Engineering | Ingestion, Cleaning & MySQL Schema | Yashaswi |
| Feature Engineering | RFM, Product, Review & Fulfillment Features | Lohit |
| Machine Learning (Person 1) | Logistic Regression Modeling & Odds Ratios | Kalyan |
| Machine Learning (Person 2) | LightGBM Modeling & Optuna Tuning | Team Member 2 |
| Machine Learning (Person 3) | Class Imbalance Experiments & Handling | Lohith Narayana |
| Evaluation & Thresholds (Person 4) | Metric Suites & Optimization Thresholds | Team Member 4 |
| Probability Calibration (Person 5) | Reliability Curves & Calibration | Team Member 5 |
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
│   └── config/
│       └── label_config.yaml             # Configurable churn observation & return windows
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
After the training pipeline finishes calibration, set the model artifact path
and version in the root `.env` file:

```env
MODEL_PATH=outputs/models/lgb_churn_model_calibrated.joblib
MODEL_VERSION=v1
```

`MODEL_PATH` may be relative to the repository root. Use the exact artifact
created by calibration if its filename differs. `MODEL_VERSION` is the label
stored with each prediction record.

---

## Running the Pipeline

The pipeline script [`scripts/run_pipeline.py`](./scripts/run_pipeline.py) orchestrates the entire workflow:

### 1. Build the database and feature tables

```bash
# Ingest the CSV files and build all feature/model-ready tables:
.venv/bin/python scripts/run_pipeline.py

# Skip ingestion when the raw tables already exist:
.venv/bin/python scripts/run_pipeline.py --skip-ingest
```

### 2. Run the complete pipeline

This command builds the feature tables, runs cohort analysis, historical trend
analysis and forecasting, trains both models, runs imbalance experiments,
performs threshold analysis and model comparison, calibrates LightGBM, and
refreshes the customer churn predictions table:

```bash
.venv/bin/python scripts/run_pipeline.py --skip-ingest --train-all
```

By default, the pipeline runs the complete analytics workflow. Individual
stages can be skipped when their inputs or outputs are already available:

```bash
.venv/bin/python scripts/run_pipeline.py --skip-cohort
.venv/bin/python scripts/run_pipeline.py --skip-trends
.venv/bin/python scripts/run_pipeline.py --skip-forecasting
.venv/bin/python scripts/run_pipeline.py --skip-predictions
```

The full pipeline creates or refreshes these analytics tables:

- `customer_cohort_analysis`
- `historical_revenue_trend`
- `historical_churn_trend`
- `historical_trend_combined`
- `forecast_revenue_trend`
- `forecast_churn_trend`
- `forecast_trend_combined`
- `churn_predictions`

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

Use `/predictions/from-features` when the caller already has feature values
and does not want the API to query MySQL. The request body has this shape:

```json
{
  "customers": [
    {
      "customer_unique_id": "C001",
      "features": {
        "monetary_value": 125.5
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

---

## Expected Table Outputs in MySQL

A successful pipeline run creates the following validated relational tables:

| Stage | Table | Description | Rows | Columns |
| :--- | :--- | :--- | :---: | :---: |
| Features | `customer_features` | Raw customer behavioral aggregations | 96,096 | 28 |
| Labels | `customer_churn_labels` | Churn flags derived from observation window | 96,095 | 5 |
| Merge | `customer_features_with_labels` | Joined features and targets | 96,095 | 32 |
| Encode | `features_encoded` | Frequency encoded & transformed features | 96,095 | 32 |
| Split | `model_ready_train` | 70% temporal train set (15 features + label + id) | 48,232 | 17 |
| Split | `model_ready_val` | 15% validation set for threshold tuning | 10,335 | 17 |
| Split | `model_ready_test` | 15% unseen test set for final reporting | 10,337 | 17 |

---

## Testing

Run the full automated test suite:
```bash
pytest
```
*Current test suite: **320/320 tests passing** across feature engineering, database pipelines, cohort analysis, class imbalance handling, Logistic Regression, and LightGBM model contracts.*

---

## License

This repository is intended for internship training and internal educational purposes.
