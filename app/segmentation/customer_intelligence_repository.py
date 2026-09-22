"""
customer_intelligence_repository.py

WHY THIS FILE EXISTS
---------------------
Risk Tier Classification and GMM Segmentation (schemas 6.1 / 6.2 of the
Module Spec) both start from the same place: one row per customer that
combines the behavioural features from the Feature Engineering module
(`features_encoded`) with the churn probability the ML module scored
into `churn_predictions` (see generate_predictions_table.py). This file
is the single place that does that merge, so Risk Tier Classification
and GMM Segmentation never each read the two tables and merge them
slightly differently.

IMPORTANT — THIS DOES NOT RE-READ features_encoded ITSELF
-------------------------------------------------------------
The ML module already has a repository function for that:
app.ml.explainability_inference.data_access.customer_feature_repository
.get_all_customer_features() — the exact function
generate_predictions_table.py itself calls to build the batch it
scores. Re-implementing a second "read features_encoded" here would
mean two independent queries against the same table that could
silently drift apart (e.g. one picks up a column rename or a dtype
change, the other doesn't, until something breaks downstream). So this
module imports and reuses that function rather than duplicating it.

The one thing that genuinely doesn't exist anywhere else yet is a
repository function for `churn_predictions` — generate_predictions_table.py
only *writes* that table (via to_sql), nothing in ml/ reads it back.
get_churn_predictions() below is new for that reason, not because it
was worth duplicating something that already existed.

WHAT IT DOES
------------
1. Gets features_encoded via the ML module's get_all_customer_features(),
   and churn_predictions via get_churn_predictions() (new — see above).
2. Merges them on customer_unique_id (inner join — a customer without a
   churn score can't be risk-tiered or fed into the churn-aware GMM, so
   rather than silently dropping/NaN-filling, we merge inner and report
   exactly who got excluded and why — not silent, same convention as
   generate_predictions_table.py's skip summary).
3. Validates the merged frame (see validate_merged_dataset) and raises
   on anything that would silently corrupt downstream tiers/clusters.

This module does not write anything — it only reads and hands back a
validated, merged pd.DataFrame. Writing the derived tables is the job
of pipeline/generate_customer_intelligence_tables.py.
"""

from __future__ import annotations

import pandas as pd

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
    """
    Merge features_encoded (via the ML module's get_all_customer_features())
    with churn_predictions (get_churn_predictions() above) on
    customer_unique_id.

    Inner join by design: Risk Tier Classification and GMM Segmentation
    both require churn_probability, so a customer who has features but
    hasn't been scored yet (or was scored but has no features — e.g. a
    stale row from a previous run) is not something either downstream
    module can do anything useful with.
    """
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
