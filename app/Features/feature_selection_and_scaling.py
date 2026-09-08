import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[1]

for candidate in (str(PROJECT_ROOT), str(APP_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from app.Database.database import engine
except ModuleNotFoundError:
    from Database.database import engine

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DB_TABLE_NAME = "features_encoded"

LABEL_COLUMN = "churn_label"
NON_FEATURE_COLUMNS = [
    "customer_unique_id", "churn_label", "censored",
    "first_purchase_date", "last_purchase_date", "reference_date",
]

TIME_SPLIT_COLUMN = "first_purchase_date"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9

def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    corr_threshold: float = 0.90,
    chi2_alpha: float = 0.05,
) -> List[str]:
    binary_cols = [
        c for c in X_train.columns
        if X_train[c].dropna().isin([0, 1]).all() and X_train[c].nunique() <= 2
    ]
    numeric_cols = [c for c in X_train.columns if c not in binary_cols]

    to_drop_corr = set()
    if len(numeric_cols) > 1:
        corr_matrix = X_train[numeric_cols].corr().abs()
        target_corr = X_train[numeric_cols].apply(lambda col: col.corr(y_train.astype(float)))
        ranked_cols = target_corr.abs().sort_values(ascending=False, kind="mergesort").index.tolist()

        kept_numeric = []
        for col in ranked_cols:
            if any(corr_matrix.loc[col, kept] > corr_threshold for kept in kept_numeric):
                to_drop_corr.add(col)
            else:
                kept_numeric.append(col)

    to_drop_chi2 = set()
    if binary_cols:
        _, p_values = chi2(X_train[binary_cols].fillna(0), y_train)
        for col, p in zip(binary_cols, p_values):
            if p >= chi2_alpha:
                to_drop_chi2.add(col)

    return [c for c in X_train.columns if c not in to_drop_corr and c not in to_drop_chi2]


def fit_scaler(X_train: pd.DataFrame) -> RobustScaler:
    scaler = RobustScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(X: pd.DataFrame, fitted: RobustScaler) -> pd.DataFrame:
    return pd.DataFrame(fitted.transform(X), columns=X.columns, index=X.index)

def load_encoded_data():
    df = pd.read_sql(f"SELECT * FROM `{DB_TABLE_NAME}`", con=engine)
    print(f"Loaded '{DB_TABLE_NAME}': {df.shape}")

    required = ["customer_unique_id", "churn_label", "censored", TIME_SPLIT_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df["customer_unique_id"].duplicated().any():
        raise ValueError("Duplicate customer_unique_id values found.")
    if df.empty:
        raise ValueError("Query returned 0 rows.")

    return df


def drop_censored(df):
    before = len(df)
    df = df[df[LABEL_COLUMN].notna()].copy()
    print(f"Dropped {before - len(df):,} censored rows. Remaining: {len(df):,}")
    return df


def time_split_data(df):
    df = df.copy()
    df[TIME_SPLIT_COLUMN] = pd.to_datetime(df[TIME_SPLIT_COLUMN])
    if df[TIME_SPLIT_COLUMN].isna().any():
        raise ValueError(f"Missing values in {TIME_SPLIT_COLUMN}.")

    df = df.sort_values(TIME_SPLIT_COLUMN, kind="mergesort").reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = train_end + int(n * VAL_FRAC)
    train_df, val_df, test_df = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    print(f"Split: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    def unpack(part):
        return part[feature_cols], part[LABEL_COLUMN].astype(int), part["customer_unique_id"]

    X_train, y_train, ids_train = unpack(train_df)
    X_val, y_val, ids_val = unpack(val_df)
    X_test, y_test, ids_test = unpack(test_df)

    sets = {"train": set(ids_train), "val": set(ids_val), "test": set(ids_test)}
    for (a, b) in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sets[a] & sets[b]
        if overlap:
            raise ValueError(f"{len(overlap)} customer(s) leaked across {a}/{b}!")

    return X_train, X_val, X_test, y_train, y_val, y_test, ids_train, ids_val, ids_test


def impute_missing(X_train, X_val, X_test):
    medians = X_train.median(numeric_only=True)
    return X_train.fillna(medians), X_val.fillna(medians), X_test.fillna(medians)


def run_scaling(X_train, X_val, X_test):
    scaler = fit_scaler(X_train)
    return apply_scaler(X_train, scaler), apply_scaler(X_val, scaler), apply_scaler(X_test, scaler)


def save_outputs(X_train, X_val, X_test, y_train, y_val, y_test, ids_train, ids_val, ids_test):
    def build(X, y, ids):
        out = X.copy()
        out.insert(0, "customer_unique_id", ids.values)
        out[LABEL_COLUMN] = y.values
        return out.round(4)

    train_out, val_out, test_out = (
        build(X_train, y_train, ids_train),
        build(X_val, y_val, ids_val),
        build(X_test, y_test, ids_test),
    )

    tables = {
        "model_ready_train": train_out,
        "model_ready_val": val_out,
        "model_ready_test": test_out,
    }

    for table_name, frame in tables.items():
        frame.to_sql(table_name, con=engine, if_exists="replace", index=False)
        print(f"Saved {table_name} to MySQL: {frame.shape}")

def main():
    df = drop_censored(load_encoded_data())

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     ids_train, ids_val, ids_test) = time_split_data(df)

    X_train, X_val, X_test = impute_missing(X_train, X_val, X_test)

    keep_cols = select_features(X_train, y_train)
    print(f"Kept {len(keep_cols)} of {X_train.shape[1]} columns")
    X_train, X_val, X_test = X_train[keep_cols], X_val[keep_cols], X_test[keep_cols]

    X_train, X_val, X_test = run_scaling(X_train, X_val, X_test)

    save_outputs(X_train, X_val, X_test, y_train, y_val, y_test, ids_train, ids_val, ids_test)
    print("Done.")


if __name__ == "__main__":
    main()
