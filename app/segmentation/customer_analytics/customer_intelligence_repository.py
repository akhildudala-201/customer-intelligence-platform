
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.Database.database import engine
from app.ml.explainibility_inference.data_access.customer_feature_repository import (
    get_all_customer_features,
)
from app.ml.explainibility_inference.inference.feature_contract import ID_COLUMN

PREDICTIONS_TABLE = "churn_predictions"
CHURN_PROBABILITY_COLUMN = "churn_probability"


def get_churn_predictions() -> pd.DataFrame:
    """One row per customer, from the ML module's generate_predictions_table.py.
    No repository function for this table exists elsewhere in the
    codebase yet — see the module docstring."""
    return pd.read_sql_table(PREDICTIONS_TABLE, con=engine)


def build_customer_intelligence_base() -> pd.DataFrame:
    features_df = get_all_customer_features()
    predictions_df = get_churn_predictions()[[ID_COLUMN, CHURN_PROBABILITY_COLUMN]]

    merged = features_df.merge(
        predictions_df,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    _report_unmatched_customers(features_df, predictions_df, merged)
    validate_merged_dataset(merged)

    return merged


def _report_unmatched_customers(
    features_df: pd.DataFrame, predictions_df: pd.DataFrame, merged: pd.DataFrame
) -> None:
    """Not fatal — printed so it's visible, same convention as
    generate_predictions_table.py's skip summary — but worth knowing
    about: customers falling out of the inner join usually mean the
    ML batch run and the feature table are out of sync."""
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
    """
    Defensive check on the merged base table before it's handed to Risk
    Tier Classification or GMM feature prep. Raises rather than letting
    a bad value quietly become a wrong tier or a distorted cluster.
    """
    if merged.empty:
        raise ValueError(
            "Merged customer_intelligence base is empty — no overlap "
            "between features_encoded (get_all_customer_features()) and "
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


def main() -> None:
    """Build and summarize the database-backed customer intelligence base."""
    merged = build_customer_intelligence_base()
if __name__ == "__main__":
    main()
