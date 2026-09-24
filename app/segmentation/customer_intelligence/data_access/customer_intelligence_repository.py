from __future__ import annotations

import pandas as pd

from app.Database.database import engine
from app.ml.explainibility_inference.data_access.customer_feature_repository import (
    get_all_customer_features,
)
from app.ml.explainibility_inference.inference.feature_contract import ID_COLUMN

PREDICTIONS_TABLE = "churn_predictions"
CUSTOMER_FEATURES_TABLE = "customer_features"
CHURN_PROBABILITY_COLUMN = "churn_probability"
RAW_GMM_FEATURE_COLUMNS = ["frequency", "recency_days"]


def get_customer_features_raw() -> pd.DataFrame:
  
    df = pd.read_sql_table(CUSTOMER_FEATURES_TABLE, con=engine)
    required = [ID_COLUMN, *RAW_GMM_FEATURE_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            f"{CUSTOMER_FEATURES_TABLE} is missing required column(s) {missing}. "
            f"Available columns: {sorted(map(str, df.columns.tolist()))}"
        )
    return df[required].copy()


def get_churn_predictions() -> pd.DataFrame:

    return pd.read_sql_table(PREDICTIONS_TABLE, con=engine)


def build_customer_intelligence_base() -> pd.DataFrame:
 
    features_df = get_all_customer_features().copy()
    features_df.columns = features_df.columns.map(str)
    predictions_df = get_churn_predictions()[[ID_COLUMN, CHURN_PROBABILITY_COLUMN]].copy()
    raw_features_df = get_customer_features_raw()


    raw_columns_to_add = [
        column for column in RAW_GMM_FEATURE_COLUMNS if column not in features_df.columns
    ]
    if raw_columns_to_add:
        raw_for_merge = raw_features_df[[ID_COLUMN, *raw_columns_to_add]]
        features_df = features_df.merge(
            raw_for_merge,
            on=ID_COLUMN,
            how="inner",
            validate="one_to_one",
        )

    merged = features_df.merge(
        predictions_df,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    merged.columns = merged.columns.map(str)

    _report_unmatched_customers(features_df, predictions_df, merged)
    validate_merged_dataset(merged)

    return merged


def _report_unmatched_customers(
    features_df: pd.DataFrame, predictions_df: pd.DataFrame, merged: pd.DataFrame
) -> None:

    features_only = set(features_df[ID_COLUMN]) - set(merged[ID_COLUMN])
    predictions_only = set(predictions_df[ID_COLUMN]) - set(merged[ID_COLUMN])

    if features_only:
        print(
            f"{len(features_only)} customer(s) have features but no churn "
            "score yet — excluded from risk tiering / segmentation until "
            "the ML module scores them."
        )
    if predictions_only:
        print(
            f"{len(predictions_only)} customer(s) have a churn score but no "
            "feature row — excluded; check for stale rows in "
            f"`{PREDICTIONS_TABLE}`."
        )


def validate_merged_dataset(merged: pd.DataFrame) -> None:
 
    if merged.empty:
        raise ValueError(
            "Merged customer_intelligence base is empty — no overlap across "
            "features_encoded, customer_features, and "
            f"`{PREDICTIONS_TABLE}`."
        )

    duplicate_ids = merged[merged.duplicated(subset=[ID_COLUMN], keep=False)]
    if not duplicate_ids.empty:
        raise ValueError(
            f"Duplicate {ID_COLUMN} values in merged dataset "
            f"({duplicate_ids[ID_COLUMN].nunique()} ids) — "
            "expected exactly one row per customer."
        )

    if merged[CHURN_PROBABILITY_COLUMN].isnull().any():
        n_null = merged[CHURN_PROBABILITY_COLUMN].isnull().sum()
        raise ValueError(f"{n_null} row(s) have a null {CHURN_PROBABILITY_COLUMN}.")

    out_of_range = merged[
        (merged[CHURN_PROBABILITY_COLUMN] < 0) | (merged[CHURN_PROBABILITY_COLUMN] > 1)
    ]
    if not out_of_range.empty:
        raise ValueError(
            f"{len(out_of_range)} row(s) have {CHURN_PROBABILITY_COLUMN} "
            "outside [0, 1] — check the ML module's calibration."
        )
