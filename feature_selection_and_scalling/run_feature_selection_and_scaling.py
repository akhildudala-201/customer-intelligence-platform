"""
run_feature_selection_and_scaling.py
======================================
PERSON F's OWN TASK, as a runnable script.

Reads customer_features_encoded.csv (produced by run_full_data_pipeline.py,
Person E's encoding step), drops censored customers (unknown true label),
splits into train/test, and applies train-only feature selection + scaling
using the reusable functions in feature_selection_and_scaling.py.

WHY THIS IS SEPARATE FROM feature_selection_and_scaling.py
-------------------------------------------------------------
feature_selection_and_scaling.py contains ONLY the three reusable,
database-free functions -- that file is what Kalyan and Kuushalie will
import directly into their own scripts. This file is different: it's
YOUR OWN driver script that loads real data, calls those functions, and
saves the final model-ready train/test sets. Kalyan and Kuushalie will
likely write their own version of this driver file that fits their own
notebooks -- what they need from you is the importable module, not this
runner.

Usage
-----
    python run_feature_selection_and_scaling.py
(edit INPUT_CSV below if your file is in a different location)
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from feature_selection_and_scaling import select_features, fit_scaler, apply_scaler


# ==========================================================
# CONFIG
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # adjust if this script lives elsewhere relative to /outputs

INPUT_CSV = PROJECT_ROOT / "outputs" / "features_encoded.csv"  # matches encoding_transformation.py's OUTPUT_PATH
OUTPUT_DIR = PROJECT_ROOT / "outputs"

LABEL_COLUMN = "churn_label"
ID_COLUMNS = ["customer_unique_id"]
# columns that are metadata, not model features (dates, censoring flag,
# reference date) -- kept aside, not fed into select_features/scaling
NON_FEATURE_COLUMNS = [
    "customer_unique_id", "churn_label", "censored",
    "first_purchase_date", "last_purchase_date", "reference_date",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42


# ==========================================================
# STEP 1: LOAD
# ==========================================================

def load_encoded_data():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"\n{INPUT_CSV} not found.\n"
            "Run run_full_data_pipeline.py first to produce this file."
        )
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {INPUT_CSV}: {df.shape}")
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
# STEP 3: TRAIN/TEST SPLIT (before any fitting!)
# ==========================================================

def split_data(df):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_cols]
    y = df[LABEL_COLUMN].astype(int)
    ids = df["customer_unique_id"]  # carried alongside, never used as a feature

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}   Test: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean()*100:.2f}%   Test churn rate: {y_test.mean()*100:.2f}%")

    # Verify no customer appears in both sets before proceeding
    overlap = set(ids_train) & set(ids_test)
    if overlap:
        raise ValueError(f"{len(overlap)} customer_unique_id(s) leaked into both train and test!")
    print(f"Customer disjointness check: OK ({len(overlap)} overlapping IDs)")

    return X_train, X_test, y_train, y_test, ids_train, ids_test


# ==========================================================
# STEP 3b: IMPUTE MISSING VALUES (train-only, same principle as scaling)
# ==========================================================

def impute_missing(X_train, X_test):
    """
    Some upstream columns (e.g. avg_delivery_days for orders that were
    never delivered) contain genuine NaNs -- not a bug, just a fact about
    those customers. Fill with the TRAIN median only, then apply that
    same value to test -- imputation is a "fit" step too, so it follows
    the exact same train-only rule as scaling.
    """
    before_train = X_train.isna().sum().sum()
    before_test = X_test.isna().sum().sum()

    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    print(f"Imputed {before_train} missing values in train, {before_test} in test (using train medians)")
    return X_train, X_test


# ==========================================================
# STEP 4: SELECT FEATURES (train-only)
# ==========================================================

def run_selection(X_train, y_train):
    keep_cols = select_features(X_train, y_train)
    print(f"select_features: kept {len(keep_cols)} of {X_train.shape[1]} columns")
    return keep_cols


# ==========================================================
# STEP 5: SCALE (fit on train, apply to both)
# ==========================================================

def run_scaling(X_train, X_test):
    scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaler(X_train, scaler)
    X_test_scaled = apply_scaler(X_test, scaler)
    return X_train_scaled, X_test_scaled, scaler


# ==========================================================
# STEP 6: SAVE MODEL-READY OUTPUT
# ==========================================================

def save_outputs(X_train_scaled, X_test_scaled, y_train, y_test, ids_train, ids_test):
    train_out = X_train_scaled.copy()
    train_out.insert(0, "customer_unique_id", ids_train.values)
    train_out[LABEL_COLUMN] = y_train.values

    test_out = X_test_scaled.copy()
    test_out.insert(0, "customer_unique_id", ids_test.values)
    test_out[LABEL_COLUMN] = y_test.values

    # Round scaled floats to 4 decimals -- does NOT change the model
    # inputs meaningfully, just stops rows from being a wall of 15-digit
    # numbers. customer_unique_id is a string, so round() skips it safely.
    train_out = train_out.round(4)
    test_out = test_out.round(4)

    train_path = OUTPUT_DIR / "model_ready_train.csv"
    test_path = OUTPUT_DIR / "model_ready_test.csv"
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)

    print(f"\nSaved: {train_path}  {train_out.shape}")
    print(f"Saved: {test_path}  {test_out.shape}")

    # Optional: a plain-text, column-aligned preview of just the first
    # rows, for quickly eyeballing the data in a text editor / terminal
    # without needing Excel. This is NOT meant to be read back into
    # pandas -- it's a human-readable snapshot only.
    preview_path = OUTPUT_DIR / "model_ready_train_preview.txt"
    with open(preview_path, "w") as f:
        f.write(train_out.head(20).to_string(index=False))
    print(f"Saved: {preview_path}  (first 20 rows, aligned columns, for viewing in a text editor)")


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("=" * 70)
    print("PERSON F -- Feature Selection & Scaling (Split-Dependent Fitting)")
    print("=" * 70)

    df = load_encoded_data()
    df = drop_censored(df)
    X_train, X_test, y_train, y_test, ids_train, ids_test = split_data(df)
    X_train, X_test = impute_missing(X_train, X_test)

    keep_cols = run_selection(X_train, y_train)
    X_train, X_test = X_train[keep_cols], X_test[keep_cols]

    X_train_scaled, X_test_scaled, scaler = run_scaling(X_train, X_test)

    # sanity checks before saving
    assert X_train_scaled.isna().sum().sum() == 0, "NaNs found in scaled train set!"
    assert X_test_scaled.isna().sum().sum() == 0, "NaNs found in scaled test set!"
    assert list(X_train_scaled.columns) == list(X_test_scaled.columns), "Train/test columns mismatch!"

    save_outputs(X_train_scaled, X_test_scaled, y_train, y_test, ids_train, ids_test)

    print("\nDone. Ready for model training.")


if __name__ == "__main__":
    main()