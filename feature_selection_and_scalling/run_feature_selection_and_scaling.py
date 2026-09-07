

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

from feature_selection_and_scaling import select_features, fit_scaler, apply_scaler


# ==========================================================
# DATABASE CONNECTION (edit to match YOUR setup)
# ==========================================================

DB_USER = "root"
DB_PASSWORD = "tiger"          # <-- your MySQL password
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "customer_sphere_db"        # <-- your database name
SOURCE_TABLE = "features_encoded"   # <-- table written by run_full_data_pipeline.py

encoded_password = quote_plus(DB_PASSWORD)
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)


# ==========================================================
# CONFIG
# ==========================================================

OUTPUT_DIR = Path(".")

LABEL_COLUMN = "churn_label"
ID_COLUMNS = ["customer_unique_id"]

# columns that are metadata, not model features (dates, censoring flag,
# reference date) -- kept aside, not fed into select_features/scaling
NON_FEATURE_COLUMNS = [
    "customer_unique_id", "churn_label", "censored",
    "first_purchase_date", "last_purchase_date", "reference_date",
]

# TEMPORARY LEAKAGE PATCH -- NOT THE REAL FIX.
# These columns are computed from a customer's FULL order history, and the
# churn label is fundamentally "did they place a repeat order" -- so these
# columns near-directly restate the label (verified: single_order_customer
# correlates 0.94 with churn_label; every model scored a perfect 1.0 with
# these included). Excluding them here stops the immediate leak, but the
# real fix is rebuilding features from first-order-only information (see
# team discussion) so the WHOLE feature set is legitimately predictive,
# not just these specific columns.
LEAKY_COLUMNS = [
    "frequency", "single_order_customer", "active_purchase_days",
    "review_count", "unique_products", "unique_categories", "total_items",
    "delivered_orders", "canceled_orders", "shipped_orders", "unavailable_orders",
    "delivered_rate", "monetary_value", "avg_order_value",
    "avg_payment_installments", "avg_delivery_days", "avg_review_score",
    "recency_days", "tenure_days",
]
NON_FEATURE_COLUMNS = NON_FEATURE_COLUMNS + LEAKY_COLUMNS

RANDOM_STATE = 42

# Time-aware split cutoffs (quantiles of first_purchase_date, among LABELED
# rows only). WHY labeled-only first: a naive split by date across ALL rows
# puts almost no labeled customers in the "recent" bucket, since recent
# customers are mostly still censored (not enough time has passed to know
# their outcome yet) -- confirmed by testing. Filtering to labeled rows
# first, then splitting by time, avoids that trap.
TRAIN_CUTOFF_QUANTILE = 0.70   # earliest 70% of labeled customers -> train
VAL_CUTOFF_QUANTILE = 0.85     # next 15% -> validation, final 15% -> test


# ==========================================================
# STEP 1: LOAD FROM DATABASE
# ==========================================================

def load_encoded_data():
    query = text(f"SELECT * FROM {SOURCE_TABLE}")
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)
    print(f"Loaded `{SOURCE_TABLE}` from MySQL: {df.shape}")
    return df


# ==========================================================
# STEP 2: DROP CENSORED ROWS (unknown true label)
# ==========================================================

def drop_censored(df):
    before = len(df)
    df = df[df[LABEL_COLUMN].notna()].copy()
    dropped = before - len(df)
    print(f"Dropped {dropped:,} censored rows (unknown label). Remaining: {len(df):,}")
    return df


# ==========================================================
# STEP 3: TIME-AWARE TRAIN / VALIDATION / TEST SPLIT
# ==========================================================

def split_data(df):
    """
    Time-aware split, matching the project's architecture diagram
    (Feature Table -> Time-aware-split -> Train/Validation/Test).

    WHY time-aware instead of random: the spec requires we never
    accidentally use future information to predict the past. Customers
    are sorted by their first_purchase_date and cut chronologically --
    earliest customers train the model, the most recent labeled
    customers become the untouched final test set. This mimics how the
    model will actually be used: trained on the past, evaluated on
    customers who came later.

    WHY a single split, not time-series cross-validation: matches the
    diagram exactly, and the minority (retained) class is small enough
    (~2,400 customers) that splitting it further across multiple CV
    folds would make per-fold metrics noisy rather than more reliable.
    """
    df = df.copy()
    df["first_purchase_date"] = pd.to_datetime(df["first_purchase_date"])
    df = df.sort_values("first_purchase_date")

    train_cutoff = df["first_purchase_date"].quantile(TRAIN_CUTOFF_QUANTILE)
    val_cutoff = df["first_purchase_date"].quantile(VAL_CUTOFF_QUANTILE)

    train_df = df[df["first_purchase_date"] < train_cutoff]
    val_df = df[(df["first_purchase_date"] >= train_cutoff) & (df["first_purchase_date"] < val_cutoff)]
    test_df = df[df["first_purchase_date"] >= val_cutoff]

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    def to_X_y(part):
        return part[feature_cols], part[LABEL_COLUMN].astype(int)

    X_train, y_train = to_X_y(train_df)
    X_val, y_val = to_X_y(val_df)
    X_test, y_test = to_X_y(test_df)

    print(f"Train cutoff date: {train_cutoff.date()}   Val cutoff date: {val_cutoff.date()}")
    print(f"Train: {X_train.shape}  (churn rate {y_train.mean()*100:.2f}%)")
    print(f"Val:   {X_val.shape}  (churn rate {y_val.mean()*100:.2f}%)")
    print(f"Test:  {X_test.shape}  (churn rate {y_test.mean()*100:.2f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ==========================================================
# STEP 3b: IMPUTE MISSING VALUES (train-only, same principle as scaling)
# ==========================================================

def impute_missing(X_train, X_val, X_test):
    """
    Some upstream columns contain genuine NaNs -- not a bug, just a fact
    about those customers. Fill with the TRAIN median only, then apply
    that same value to validation and test -- imputation is a "fit" step
    too, so it follows the exact same train-only rule as scaling.
    """
    before = X_train.isna().sum().sum() + X_val.isna().sum().sum() + X_test.isna().sum().sum()

    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_val = X_val.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    print(f"Imputed {before} missing values total (using train medians only)")
    return X_train, X_val, X_test


# ==========================================================
# STEP 4: SELECT FEATURES (train-only)
# ==========================================================

def run_selection(X_train, y_train):
    keep_cols = select_features(X_train, y_train)
    print(f"select_features: kept {len(keep_cols)} of {X_train.shape[1]} columns")
    return keep_cols


# ==========================================================
# STEP 5: SCALE (fit on train, apply to all three)
# ==========================================================

def run_scaling(X_train, X_val, X_test):
    scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaler(X_train, scaler)
    X_val_scaled = apply_scaler(X_val, scaler)
    X_test_scaled = apply_scaler(X_test, scaler)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ==========================================================
# STEP 6: SAVE MODEL-READY OUTPUT
# ==========================================================

def save_outputs(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test):
    train_out = X_train_scaled.copy()
    train_out[LABEL_COLUMN] = y_train.values
    val_out = X_val_scaled.copy()
    val_out[LABEL_COLUMN] = y_val.values
    test_out = X_test_scaled.copy()
    test_out[LABEL_COLUMN] = y_test.values

    train_path = OUTPUT_DIR / "model_ready_train.csv"
    val_path = OUTPUT_DIR / "model_ready_val.csv"
    test_path = OUTPUT_DIR / "model_ready_test.csv"
    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)
    test_out.to_csv(test_path, index=False)

    print(f"\nSaved: {train_path}  {train_out.shape}")
    print(f"Saved: {val_path}  {val_out.shape}")
    print(f"Saved: {test_path}  {test_out.shape}")


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("=" * 70)
    print("PERSON F -- Feature Selection & Scaling (Split-Dependent Fitting)")
    print("Using TIME-AWARE train/validation/test split")
    print("=" * 70)

    df = load_encoded_data()
    df = drop_censored(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train, X_val, X_test = impute_missing(X_train, X_val, X_test)

    keep_cols = run_selection(X_train, y_train)
    X_train, X_val, X_test = X_train[keep_cols], X_val[keep_cols], X_test[keep_cols]

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = run_scaling(X_train, X_val, X_test)

    # sanity checks before saving
    assert X_train_scaled.isna().sum().sum() == 0, "NaNs found in scaled train set!"
    assert X_val_scaled.isna().sum().sum() == 0, "NaNs found in scaled validation set!"
    assert X_test_scaled.isna().sum().sum() == 0, "NaNs found in scaled test set!"
    assert list(X_train_scaled.columns) == list(X_val_scaled.columns) == list(X_test_scaled.columns), \
        "Train/val/test columns mismatch!"

    save_outputs(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)

    print("\nDone. Ready for model training.")


if __name__ == "__main__":
    main()