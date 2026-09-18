# Customer Intelligence Platform

A production backend platform for customer intelligence — churn prediction, customer segmentation, and targeted retention marketing — built on the Olist Brazilian e-commerce dataset.

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

## Engineered Behavioral Features (14 Features)

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
| 14 | `preferred_payment_type_not_defined` | Binary (0/1) | One-hot encoded payment preference: Not Defined / voucher fallback |

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
python3 scripts/run_pipeline.py

# Skip CSV ingestion if raw tables already exist in MySQL:
python3 scripts/run_pipeline.py --skip-ingest
```

### Option B: Build Tables and Train Models
```bash
# Build tables and train Logistic Regression:
python3 scripts/run_pipeline.py --skip-ingest --train-logistic

# Build tables and train LightGBM:
python3 scripts/run_pipeline.py --skip-ingest --train-lightgbm

# Build tables and train BOTH models:
python3 scripts/run_pipeline.py --skip-ingest --train-all
```

### Option C: Run Standalone ML Modules Directly
```bash
# Run standalone Logistic Regression training:
python3 app/ml/logistic_regression.py

# Run standalone LightGBM training (fast mode ~3s):
python3 app/ml/train_lightgbm_model.py

# Run Class Imbalance 8-strategy benchmarks:
python3 app/ml/imbalance_experiments.py
```

---

## Expected Table Outputs in MySQL

A successful pipeline run creates the following validated relational tables:

| Stage | Table | Description | Rows | Columns |
| :--- | :--- | :--- | :---: | :---: |
| Features | `customer_features` | Raw customer behavioral aggregations | 96,096 | 28 |
| Labels | `customer_churn_labels` | Churn flags derived from observation window | 96,095 | 5 |
| Merge | `customer_features_with_labels` | Joined features and targets | 96,095 | 32 |
| Encode | `features_encoded` | Frequency encoded & transformed features | 96,095 | 32 |
| Split | `model_ready_train` | 70% temporal train set (14 features + label + id) | 48,232 | 16 |
| Split | `model_ready_val` | 15% validation set for threshold tuning | 10,335 | 16 |
| Split | `model_ready_test` | 15% unseen test set for final reporting | 10,337 | 16 |

---

## Testing

Run the full automated test suite:
```bash
pytest
```
*Current test suite: **164/164 tests passing** with 0 warnings across feature engineering, database pipelines, class imbalance handling, Logistic Regression, and LightGBM model contracts.*

---

## License

This repository is intended for internship training and internal educational purposes.
