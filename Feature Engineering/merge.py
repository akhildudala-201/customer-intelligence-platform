from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_MODULE_DIR = PROJECT_ROOT / "app" / "feature_engineering"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FEATURE_MODULE_DIR))

from build_labels import build_labels
from app.feature_engineering.build_features import (
    build_customer_base,
    build_fulfillment_features,
    build_geography_features,
    build_order_features,
    build_payment_features,
    build_product_features,
    build_review_features,
    build_rfm_features,
    build_time_features,
    clean_features,
    engine,
    merge_features,
    validate_features,
)


OUTPUT_PATH = PROJECT_ROOT / "outputs" / "customer_features_with_labels.csv"
TABLE_NAME = "customer_features_with_labels"


def build_combined_dataset():
    labels, reference_date = build_labels()
    customer_base = build_customer_base(reference_date)

    feature_tables = {
        "rfm": build_rfm_features(reference_date),
        "order_behavior": build_order_features(reference_date),
        "payment": build_payment_features(reference_date),
        "reviews": build_review_features(reference_date),
        "products": build_product_features(reference_date),
        "fulfillment": build_fulfillment_features(reference_date),
        "time_behavior": build_time_features(reference_date),
        "geography": build_geography_features(reference_date),
    }

    features = merge_features(customer_base, feature_tables)
    features["reference_date"] = reference_date
    features = clean_features(features)
    features = validate_features(features, customer_base)

    combined = features.merge(
        labels[["customer_unique_id", "label", "censored"]],
        on="customer_unique_id",
        how="inner",
        validate="one_to_one",
    )
    combined = combined.rename(columns={"label": "churn_label"})

    if len(combined) != len(labels):
        raise ValueError(
            "Some churn-label records could not be matched to a feature record."
        )

    return combined


def save_combined_dataset(dataset: pd.DataFrame):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    dataset.to_sql(TABLE_NAME, con=engine, if_exists="replace", index=False)

    print(f"\nCombined CSV saved to: {OUTPUT_PATH}")
    print(f"Combined database table replaced: {TABLE_NAME}")
    print(f"Final shape: {dataset.shape}")


def main():
    combined = build_combined_dataset()
    save_combined_dataset(combined)


if __name__ == "__main__":
    main()
