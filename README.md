# Customer Intelligence Platform

A backend platform for customer intelligence — churn prediction, customer segmentation, and targeted marketing — built on the Olist Brazilian e-commerce dataset.

This repository implements the end-to-end data pipeline and machine learning layer: CSV ingestion → cleaned MySQL tables → engineered behavioral features → churn labels → encoded & scaled model-ready splits → class imbalance handling → high-performance ML models (LightGBM & Logistic Regression) with automated experiment tracking.

---

## Project Overview

Using the Olist E-Commerce dataset, the platform:

- Cleans and loads 8 raw CSVs into relational MySQL tables.
- Engineers 14 non-leaking, high-impact behavioral features (checkout freight burden, product weight, RFM monetary spend, review engagement, logistics delay, category frequencies).
- Derives customer churn labels using a configurable return-window rule (`label_config.yaml`).
- Encodes and scales features into time-split `model_ready_train`, `model_ready_val`, and `model_ready_test` tables.
- Benchmarks 8 class imbalance strategies (cost-sensitive loss weighting vs. resampling) to handle the 30.26:1 churn skew.
- Trains production-grade churn classifiers:
  - **LightGBM**: Bayesian hyperparameter-tuned tree ensemble achieving **`0.9492 ROC-AUC`** and **`13.2x Lift`** on top 5% risk slice.
  - **Logistic Regression**: High-precision baseline achieving **`0.8770 ROC-AUC`** and **`80.61% Balanced Accuracy`** with odds-ratio interpretability.
- Automatically logs all training runs, thresholds, and confusion matrix metrics to centralized experiment trackers.

---

## Model Performance Benchmarks

Evaluated on the unseen test set (10,337 customers, 5.82% minority base rate):

| Model | ROC-AUC | PR-AUC (Avg Prec) | Operating Threshold | Precision | Recall | Balanced Acc | Lift over Random | Primary Use Case |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LightGBM** | **`0.9492`** | **`0.7661`** | `Rate Mode (Top 5%)` | **`74.85%`** | **`64.29%`** | — | **`12.85x`** | **Automated Retention Targeting** |
| **Logistic Regression** | **`0.8770`** | **`0.9892`** | `0.59 (Val-tuned)` | **`99.19%`** | **`70.53%`** | **`80.61%`** | — | **Auditable Strategic Insights** |

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
| **Testing** | Pytest | 164 unit and integration tests |

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
│   │   └── experiment_logger.py          # Centralized CSV/Markdown experiment tracker
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
├── tests/                                # 164 unit tests (database, features, ML models)
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
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
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
Ensure `DB_USER`, `DB_PASSWORD`, `DB_NAME=olist`, and `DATASET_DIR` are populated.

---

## Running the Pipeline

The pipeline script [`scripts/run_pipeline.py`](./scripts/run_pipeline.py) orchestrates the entire workflow:

### Option A: Build Database & Feature Tables Only
```bash
# Full run including raw CSV ingestion:
python scripts/run_pipeline.py

# Skip CSV ingestion if raw tables already exist in MySQL:
python scripts/run_pipeline.py --skip-ingest
```

### Option B: Build Tables and Train Models
```bash
# Build tables and train Logistic Regression:
python scripts/run_pipeline.py --skip-ingest --train-logistic

# Build tables and train LightGBM:
python scripts/run_pipeline.py --skip-ingest --train-lightgbm

# Build tables and train BOTH models:
python scripts/run_pipeline.py --skip-ingest --train-all
```

### Option C: Run Standalone ML Modules Directly
```bash
# Run standalone Logistic Regression training:
python app/ml/logistic_regression.py

# Run standalone LightGBM training (fast mode ~3s):
python app/ml/train_lightgbm_model.py

# Run Class Imbalance 8-strategy benchmarks:
python app/ml/imbalance_experiments.py
```

---

## Expected Table Outputs in MySQL

A successful run creates the following validated tables:

| Stage | Table | Description | Rows |
| :--- | :--- | :--- | :---: |
| Features | `customer_features` | Raw customer aggregations | 96,096 |
| Labels | `customer_churn_labels` | Churn flags based on observation window | 96,095 |
| Merge | `customer_features_with_labels` | Joined features and targets | 96,095 |
| Encode | `features_encoded` | Frequency encoded & transformed features | 96,095 |
| Split | `model_ready_train` | 70% temporal train set (14 features) | 48,232 |
| Split | `model_ready_val` | 15% validation set for threshold tuning | 10,335 |
| Split | `model_ready_test` | 15% unseen test set for final reporting | 10,337 |

---

## Testing

Run the comprehensive test suite:
```bash
pytest
```
*Current coverage: **164/164 tests passing** across feature engineering, database pipelines, class imbalance handling, Logistic Regression, and LightGBM model contracts.*

---

## License

This repository is intended for internship training and internal educational purposes.
