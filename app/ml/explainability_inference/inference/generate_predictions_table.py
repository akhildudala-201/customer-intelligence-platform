from __future__ import annotations

import json

import pandas as pd

from app.ml.explainability_inference.data_access.customer_feature_repository import (
    get_all_customer_features,
)
from app.ml.explainability_inference.inference.feature_contract import (
    ID_COLUMN,
    REQUIRED_FEATURES,
)
from app.ml.explainability_inference.inference.predict import ChurnPredictor

OUTPUT_TABLE_NAME = "churn_predictions"


def _split_scoreable_rows(features_df: pd.DataFrame) -> pd.DataFrame:
   
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

    features_df = get_all_customer_features()
    scoreable_df = _split_scoreable_rows(features_df)

    predictor = ChurnPredictor()
    records = predictor.predict(scoreable_df)
    output_df = _records_to_dataframe(records)

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
