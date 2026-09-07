
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from feature_selection_and_scaling import select_features, fit_scaler, apply_scaler
from database import engine

# ==========================================================
# CONFIG
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # adjust if this script lives elsewhere relative to /outputs

INPUT_CSV = PROJECT_ROOT / "outputs" / "features_encoded.csv"  # matches encoding_transformation.py's OUTPUT_PATH
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Set to True to load from MySQL instead of the CSV above.
USE_DATABASE = True
DB_TABLE_NAME = "features_encoded"

# NOTE: DB credentials are no longer hardcoded here. `from database import
# engine` above already builds a connection using DB_HOST/DB_PORT/DB_USER/
# DB_PASSWORD/DB_NAME read from your .env file (see database.py). Update
# your .env if the credentials change -- never put them back in this file.

LABEL_COLUMN = "churn_label"
ID_COLUMNS = ["customer_unique_id"]
# columns that are metadata, not model features (dates, censoring flag,
# reference date) -- kept aside, not fed into select_features/scaling
NON_FEATURE_COLUMNS = [
    "customer_unique_id", "churn_label", "censored",
    "first_purchase_date", "last_purchase_date", "reference_date",
]

# --- TIME-AWARE SPLIT CONFIG ---
# Which date column determines chronological order. first_purchase_date
# sorts customers by WHEN THEY JOINED (train on earlier cohorts, test on
# newer ones -- good default for "will a new customer churn?"). Switch to
# last_purchase_date if you'd rather split by most recent activity instead.
TIME_SPLIT_COLUMN = "first_purchase_date"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9, "split fractions must sum to 1.0"


# ==========================================================
# STEP 1: LOAD
# ==========================================================

def load_encoded_data():
    query = f"""
        SELECT *
        FROM `{DB_TABLE_NAME}`
    """

    df = pd.read_sql(query, con=engine)

    print(f"Loaded table '{DB_TABLE_NAME}' from MySQL: {df.shape}")

    # Existing input validation remains unchanged
    required_cols = [
        "customer_unique_id",
        "churn_label",
        "censored",
        TIME_SPLIT_COLUMN,
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"DB table '{DB_TABLE_NAME}' is missing required columns: {missing}"
        )

    if df["customer_unique_id"].duplicated().any():
        raise ValueError(
            f"DB table '{DB_TABLE_NAME}' has duplicate customer_unique_id values."
        )

    if df.empty:
        raise ValueError(
            f"DB table '{DB_TABLE_NAME}' returned 0 rows."
        )

    print("DB input validation: OK")

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

def time_split_data(df):
    """
    Sort customers chronologically by TIME_SPLIT_COLUMN, then cut the
    timeline into three consecutive blocks -- no shuffling. Earliest
    TRAIN_FRAC of customers -> train, next VAL_FRAC -> validation, most
    recent TEST_FRAC -> test.
    """
    df = df.copy()
    df[TIME_SPLIT_COLUMN] = pd.to_datetime(df[TIME_SPLIT_COLUMN])

    if df[TIME_SPLIT_COLUMN].isna().any():
        n_missing = df[TIME_SPLIT_COLUMN].isna().sum()
        raise ValueError(
            f"{n_missing} rows have a missing {TIME_SPLIT_COLUMN} -- "
            "cannot chronologically order them. Investigate before splitting."
        )

    df = df.sort_values(TIME_SPLIT_COLUMN, kind="mergesort").reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = train_end + int(n * VAL_FRAC)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(
        f"Time-aware split by '{TIME_SPLIT_COLUMN}':\n"
        f"  Train: {len(train_df):,} rows  "
        f"({train_df[TIME_SPLIT_COLUMN].min().date()} to {train_df[TIME_SPLIT_COLUMN].max().date()})\n"
        f"  Val:   {len(val_df):,} rows  "
        f"({val_df[TIME_SPLIT_COLUMN].min().date()} to {val_df[TIME_SPLIT_COLUMN].max().date()})\n"
        f"  Test:  {len(test_df):,} rows  "
        f"({test_df[TIME_SPLIT_COLUMN].min().date()} to {test_df[TIME_SPLIT_COLUMN].max().date()})"
    )

    def churn_rate(part):
        return part[LABEL_COLUMN].astype(int).mean() * 100

    print(
        f"Churn rate -- Train: {churn_rate(train_df):.2f}%   "
        f"Val: {churn_rate(val_df):.2f}%   "
        f"Test: {churn_rate(test_df):.2f}%  "
        f"(these will differ from each other -- that's expected with a time split, not stratified)"
    )

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    def unpack(part):
        X = part[feature_cols]
        y = part[LABEL_COLUMN].astype(int)
        ids = part["customer_unique_id"]
        return X, y, ids

    X_train, y_train, ids_train = unpack(train_df)
    X_val, y_val, ids_val = unpack(val_df)
    X_test, y_test, ids_test = unpack(test_df)

    # Disjointness check across all three sets (should be guaranteed by
    # construction, but verify rather than assume -- same principle as
    # the earlier random-split leakage check)
    train_ids, val_ids, test_ids = set(ids_train), set(ids_val), set(ids_test)
    overlaps = {
        "train/val": train_ids & val_ids,
        "train/test": train_ids & test_ids,
        "val/test": val_ids & test_ids,
    }
    for pair, overlap in overlaps.items():
        if overlap:
            raise ValueError(f"{len(overlap)} customer_unique_id(s) leaked across {pair}!")
    print("Customer disjointness check: OK (0 overlapping IDs across all three sets)")

    return X_train, X_val, X_test, y_train, y_val, y_test, ids_train, ids_val, ids_test


# ==========================================================
# STEP 3b: IMPUTE MISSING VALUES (train-only, same principle as scaling)
# ==========================================================

def impute_missing(X_train, X_val, X_test):
    """
    Some upstream columns (e.g. avg_delivery_days for orders that were
    never delivered) contain genuine NaNs -- not a bug, just a fact about
    those customers. Fill with the TRAIN median only, then apply that
    same value to val/test -- imputation is a "fit" step too, so it
    follows the exact same train-only rule as scaling.
    """
    before_train = X_train.isna().sum().sum()
    before_val = X_val.isna().sum().sum()
    before_test = X_test.isna().sum().sum()

    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_val = X_val.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    print(
        f"Imputed {before_train} missing values in train, {before_val} in val, "
        f"{before_test} in test (using train medians)"
    )
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

def save_outputs(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test,
                  ids_train, ids_val, ids_test):

    def build(X_scaled, y, ids):
        out = X_scaled.copy()
        out.insert(0, "customer_unique_id", ids.values)
        out[LABEL_COLUMN] = y.values
        return out.round(4)

    train_out = build(X_train_scaled, y_train, ids_train)
    val_out = build(X_val_scaled, y_val, ids_val)
    test_out = build(X_test_scaled, y_test, ids_test)

    train_path = OUTPUT_DIR / "model_ready_train.csv"
    val_path = OUTPUT_DIR / "model_ready_val.csv"
    test_path = OUTPUT_DIR / "model_ready_test.csv"

    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)
    test_out.to_csv(test_path, index=False)

    print(f"\nSaved: {train_path}  {train_out.shape}")
    print(f"Saved: {val_path}  {val_out.shape}")
    print(f"Saved: {test_path}  {test_out.shape}")

    preview_path = OUTPUT_DIR / "model_ready_train_preview.txt"
    with open(preview_path, "w") as f:
        f.write(train_out.head(20).to_string(index=False))
    print(f"Saved: {preview_path}  (first 20 rows, aligned columns, for viewing in a text editor)")


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("=" * 70)
    print("PERSON F -- Feature Selection & Scaling (Time-Aware Split)")
    print("=" * 70)

    df = load_encoded_data()
    df = drop_censored(df)

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     ids_train, ids_val, ids_test) = time_split_data(df)

    X_train, X_val, X_test = impute_missing(X_train, X_val, X_test)

    keep_cols = run_selection(X_train, y_train)
    X_train, X_val, X_test = X_train[keep_cols], X_val[keep_cols], X_test[keep_cols]

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = run_scaling(X_train, X_val, X_test)

    # sanity checks before saving
    assert X_train_scaled.isna().sum().sum() == 0, "NaNs found in scaled train set!"
    assert X_val_scaled.isna().sum().sum() == 0, "NaNs found in scaled val set!"
    assert X_test_scaled.isna().sum().sum() == 0, "NaNs found in scaled test set!"
    assert list(X_train_scaled.columns) == list(X_val_scaled.columns) == list(X_test_scaled.columns), \
        "Train/val/test columns mismatch!"

    save_outputs(X_train_scaled, X_val_scaled, X_test_scaled,
                 y_train, y_val, y_test, ids_train, ids_val, ids_test)

    print("\nDone. Ready for model training.")


if __name__ == "__main__":
    main()