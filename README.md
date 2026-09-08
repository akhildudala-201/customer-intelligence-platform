# Customer Intelligence Platform

A backend platform for customer intelligence — churn prediction, customer segmentation, and targeted marketing — built on the Olist Brazilian e-commerce dataset.

This repo currently implements the **data engineering and feature engineering pipeline**: CSV → cleaned MySQL tables → engineered features → churn labels → encoded, scaled, train/val/test-split model-ready tables. The ML training, segmentation/campaign, and FastAPI layers are planned but not yet in this codebase (see [Current Status](#current-status)).


---

## Project Overview

Using the Olist E-Commerce dataset, the pipeline:

- Cleans and loads 8 raw CSVs into MySQL
- Engineers ~28 customer-level behavioral features (RFM, order behavior, payments, reviews, products, fulfillment, geography, time)
- Derives a churn label from a configurable return-window rule
- Encodes and transforms features (frequency encoding, one-hot encoding, log1p)
- Selects/scales features and produces time-based `train` / `val` / `test` tables in MySQL



---

## Tech Stack

**Implemented and in `requirements.txt`:**

| Tool | Used for |
|---|---|
| Python 3.11+ | Runtime (tested on 3.12) |
| Pandas / NumPy | Data cleaning & feature engineering |
| SQLAlchemy + PyMySQL | MySQL connectivity |
| scikit-learn | Feature selection (chi2) & scaling (RobustScaler) |
| PyYAML | Churn-label configuration |
| python-dotenv | Loading `.env` |

---

## Team

| Module | Owner |
|---|---|
| Data Engineering | Yashaswi |
| Feature Engineering | Lohit |
| Machine Learning | Kalyan |
| Segmentation & Campaigns | Kuushalie |
| FastAPI & Integration | Rajeswari |

---

## Repository Structure

This is the actual current layout (verified against the repo, not aspirational):

```
customer-intelligence-platform/
├── .github/
│   └── PULL_REQUEST_TEMPLATE.md
├── app/
│   ├── Database/
│   │   ├── cleaning.py      
│   │   ├── database.py       
│   │   └── ingest.py         
│   ├── Features/
│   │   ├── build_features.py             
│   │   ├── build_churn_label.py          
│   │   ├── merge.py                      
│   │   ├── encoding_transformation.py    
│   │   └── feature_selection_and_scaling.py 
│   └── config/
│       └── label_config.yaml       # churn-label rules 
├── data/
│   ├── README.md             # dataset documentation (tables, joins, churn-label ideas)
│   └── olist_*.csv           # 8 raw source CSVs
├── docs/                    
├── tests/                    
├── scripts/
│   └── run_pipeline.py       # orchestrates the full pipeline end-to-end
├── run_pipeline.py           # thin wrapper: `python run_pipeline.py` at repo root
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── Makefile                  
└── requirements.txt
```

---

## Prerequisites

- Python 3.11 or newer
- MySQL 8.x or MariaDB 10.11+, installed and running locally (or reachable over the network)
- Git

---

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd customer-intelligence-platform
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up MySQL

Create the database and apply the schema **before running ingestion** — this matters. If you skip this step, `pandas.to_sql` will silently auto-create tables for you with no primary keys, no foreign keys, and generic types (e.g. every ID column becomes `TEXT`), which is a materially weaker schema than the one below.

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS olist CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
mysql -u root -p olist < data/Schema.sql
```

> `data/Schema.sql` defines all foreign keys, indexes, and primary keys for the tables in `olist`.

### 5. Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env` and fill in your real values — at minimum `DB_USER`, `DB_PASSWORD`, and `DATASET_DIR` (absolute path to this repo's `data/` folder on your machine). `database.py` reads `DB_HOST` / `DB_PORT` / `DB_USER` / `DB_PASSWORD` / `DB_NAME` individually to build the connection — it does **not** read `DATABASE_URL` directly, so keep the two in sync manually if you use both.


### 6. Run the pipeline

Option A — run everything in one command from the repo root:

```bash
python run_pipeline.py
```

This runs, in order: ingest all CSVs (`--replace`) → build features → build churn labels → merge → encode/transform → select & scale/split. If your data is already ingested and you just want to re-run feature engineering:

```bash
python run_pipeline.py --skip-ingest
```

Option B — run each stage individually (useful while debugging one step):

```bash
python -m app.Database.ingest all --replace
python app/Features/build_features.py
python app/Features/build_churn_label.py
python app/Features/merge.py
python app/Features/encoding_transformation.py
python app/Features/feature_selection_and_scaling.py
```

### Expected result

A full run against the unmodified dataset produces these row counts — useful as a sanity check that your setup is correct:

| Stage | Table | Rows |
|---|---|---|
| Ingest | `customers` | 99,441 |
| Ingest | `orders` | 99,441 |
| Ingest | `order_items` | 112,633 |
| Ingest | `order_payments` | 103,886 |
| Ingest | `order_reviews` | 100,000 |
| Ingest | `products` | 32,950 |
| Ingest | `sellers` | 3,095 |
| Ingest | `geolocation` | 1,000,163 |
| Features | `customer_features` | 96,096 |
| Labels | `customer_churn_labels` | 96,095 |
| Merge | `customer_features_with_labels` | 96,095 |
| Encode | `features_encoded` | 96,095 |
| Split | `model_ready_train` | 48,232 |
| Split | `model_ready_val` | 10,335 |
| Split | `model_ready_test` | 10,337 |

The split step drops ~27,191 "censored" customers (not enough time elapsed since their last order to confidently label them churned/retained under the 180-day window in `label_config.yaml`) before splitting the rest 70/15/15 by first-purchase date.

Ingestion is the slow step (the geolocation CSV alone is ~1M rows, processed in chunks); the feature/label/merge/encode/scale steps each complete in well under a minute on the full dataset.


---

## Development Workflow

1. Pull the latest changes from `main`
2. Create a feature branch
3. Implement your changes
4. Test locally
5. Commit your changes
6. Push your branch
7. Open a Pull Request
8. Wait for code review before merging

## Branch Naming

```
feature/<feature-name>
bugfix/<bug-name>
docs/<document-name>
```

Examples: `feature/churn-model`, `feature/customer-api`, `bugfix/sqlite-join`

## Commit Message Convention

```
feat: add churn prediction endpoint
fix: resolve SQLite foreign key issue
docs: update README
refactor: simplify feature engineering
test: add API unit tests
```

Full contribution guidelines (PR checklist, coding standards, module ownership) are in [`CONTRIBUTING.md`](./CONTRIBUTING.md).

---

## Testing

```bash
pytest
```


---

## Documentation

- [`data/README.md`](./data/README.md) — dataset documentation: table shapes, join keys, and how the Olist data maps to our real membership data
- `docs/` — reserved for additional project documentation 

---

## Contributing

Please read [`CONTRIBUTING.md`](./CONTRIBUTING.md) before submitting a Pull Request.

---

## License

This repository is intended for internship training and internal learning purposes.
