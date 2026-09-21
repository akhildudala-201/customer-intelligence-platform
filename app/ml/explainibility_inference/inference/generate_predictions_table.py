"""
generate_predictions_table.py

WHY THIS FILE EXISTS
---------------------
`ChurnPredictor.predict()` and the FastAPI routes in api/main.py answer
one customer (or one requested batch) at a time — useful on demand, but
the segmentation module needs a full, queryable table: every customer's
latest churn_probability, reason_codes, etc., in one place. This script
is the final step of the ML part's pipeline: it scores the whole
`features_encoded` table in one pass and writes the result to a new
`churn_predictions` table, which is what segmentation reads from. It
does not touch `features_encoded` or any other existing table — only
creates/replaces `churn_predictions`.

WHAT IT DOES
------------
1. Reads every customer's feature row (data_access.get_all_customer_features).
2. Splits out any row with a null in a REQUIRED_FEATURES column (real
   data gaps — e.g. avg_delivery_days is null for orders that were
   never delivered — not something feature_contract.validate_features()
   should silently paper over) and skips those rows rather than failing
   the whole batch, printing which customers/columns were skipped so
   it's visible, not silent.
3. Runs the full inference pipeline once per remaining customer via
   ChurnPredictor (same model load, SHAP explainer, reason-code logic
   used everywhere else in this package — no separate/duplicate scoring
   logic here; validate_features() inside predict() still runs as the
   final defensive check, same as every other call site).
4. Writes the results to the `churn_predictions` MySQL table, replacing
   whatever was there before (`if_exists="replace"`) — this table is
   always a full, current snapshot, not an append-only log. shap_values
   and reason_codes are JSON-encoded columns (plain dict/list objects
   aren't valid SQL column types).

RUNNING IT
----------
    python -m app.ml.explainibility_inference.inference.generate_predictions_table

Requires the same env vars as the rest of the package: DB_HOST/DB_PORT/
DB_USER/DB_PASSWORD/DB_NAME (for reading+writing) and MODEL_PATH/
MODEL_VERSION (for ChurnPredictor). No dummy fallback for any of these,
consistent with predict.py.
"""

from __future__ import annotations

import json

import pandas as pd

from app.ml.explainibility_inference.data_access.customer_feature_repository import (
    get_all_customer_features,
)
from app.ml.explainibility_inference.inference.feature_contract import (
    ID_COLUMN,
    REQUIRED_FEATURES,
)
from app.ml.explainibility_inference.inference.predict import ChurnPredictor

OUTPUT_TABLE_NAME = "churn_predictions"


def _split_scoreable_rows(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Separate customers with a complete feature row from ones missing a
    required value, instead of letting one incomplete row fail the
    entire batch (validate_features() inside predict() would reject the
    whole DataFrame on the first null it finds).

    Returns only the scoreable rows. Prints a summary of what got
    skipped and why — not silent, just not fatal.
    """
    null_mask = features_df[REQUIRED_FEATURES].isnull().any(axis=1)
    skipped = features_df[null_mask]
    scoreable = features_df[~null_mask].reset_index(drop=True)

    if not skipped.empty:
        null_counts = skipped[REQUIRED_FEATURES].isnull().sum()
        null_counts = null_counts[null_counts > 0]
        print(
            f"Skipping {len(skipped)} customer(s) with missing required "
            f"feature value(s): {null_counts.to_dict()}"
        )

    if scoreable.empty:
        raise RuntimeError(
            f"No customers had complete data for all {len(REQUIRED_FEATURES)} "
            "required features — nothing to score. Check `features_encoded` "
            "for missing values (see the printed skip summary above)."
        )

    return scoreable


def _records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Flatten predict() output into SQL-writable columns (dict/list ->
    JSON strings; everything else is already a plain scalar)."""
    rows = []
    for record in records:
        rows.append(
            {
                "customer_unique_id": record["customer_unique_id"],
                "churn_probability": record["churn_probability"],
                "shap_values": json.dumps(record["shap_values"]),
                "reason_codes": json.dumps(record["reason_codes"]),
                "model_version": record["model_version"],
                "scored_at": record["scored_at"],
            }
        )
    return pd.DataFrame(rows)


def generate_predictions_table() -> pd.DataFrame:
    """
    Score every customer in `features_encoded` with complete data and
    overwrite the `churn_predictions` table with the results. Returns
    the written DataFrame (useful for the smoke check / a quick local
    look, without having to re-query the table).
    """
    features_df = get_all_customer_features()
    scoreable_df = _split_scoreable_rows(features_df)

    predictor = ChurnPredictor()
    records = predictor.predict(scoreable_df)
    output_df = _records_to_dataframe(records)

    # Imported lazily, right before it's needed: get_all_customer_features()
    # already needs DB_* env vars to actually read, but ordering the
    # import here means a "no customers" / prediction failure surfaces as
    # that specific error rather than a misleading missing-env-var one.
    from app.Database.database import engine

    output_df.to_sql(
        OUTPUT_TABLE_NAME,
        con=engine,
        if_exists="replace",
        index=False,
    )
    return output_df


if __name__ == "__main__":
    result = generate_predictions_table()
    print(f"Wrote {len(result)} rows to `{OUTPUT_TABLE_NAME}`.")