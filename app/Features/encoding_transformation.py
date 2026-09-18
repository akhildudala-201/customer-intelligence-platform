import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[1]

for candidate in (str(PROJECT_ROOT), str(APP_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from app.Database.database import engine
except ModuleNotFoundError:
    from Database.database import engine

INPUT_DATABASE_TABLE = "customer_features_with_labels"
DATABASE_TABLE = "features_encoded"

ONE_HOT_COLUMNS = [
    "latest_order_status",
    "preferred_payment_type",
]

FREQUENCY_ENCODE_COLUMNS = [
    "dominant_product_category",
    "customer_city_state",
]

LOG1P_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary_value",
    "avg_order_value",
    "delivered_orders",
    "canceled_orders",
    "shipped_orders",
    "unavailable_orders",
    "review_count",
    "total_items",
    "unique_products",
    "unique_categories",
    "active_purchase_days",
]

def load_data():
    """
    Load the customer-level dataset directly from MySQL.

    The input table replaces the CSV file that was
    previously used by the pipeline.
    """

    print("\nLoading input dataset from MySQL...")
    print("-" * 60)

    try:

        query = f"""
            SELECT *
            FROM {INPUT_DATABASE_TABLE}
        """

        df = pd.read_sql(
            query,
            con=engine
        )

    except Exception as error:

        print(
            "\nERROR: Failed to load input data "
            "from MySQL."
        )

        raise error

    if df.empty:

        raise ValueError(
            f"MySQL input table '{INPUT_DATABASE_TABLE}' "
            f"is empty."
        )

    print(
        "\nInput dataset loaded successfully "
        "from MySQL."
    )

    print(
        f"Input table : {INPUT_DATABASE_TABLE}"
    )

    print(
        f"Input shape : {df.shape}"
    )

    return df

def frequency_encode(df):
    df = df.copy()

    print("\nApplying Frequency Encoding...")
    print("-" * 60)

    for column in FREQUENCY_ENCODE_COLUMNS:

        if column not in df.columns:

            raise ValueError(
                f"Column not found for frequency encoding: "
                f"{column}"
            )

        # Handle missing values
        df[column] = df[column].fillna("unknown")
        
        frequency_map = (
            df[column]
            .value_counts(normalize=True)
        )

        # New encoded column
        encoded_column = (
            f"{column}_frequency"
        )

        df[encoded_column] = (
            df[column]
            .map(frequency_map)
            .fillna(0)
        )

        df.drop(
            columns=[column],
            inplace=True
        )

        print(
            f"  {column} -> {encoded_column}"
        )

    return df

def one_hot_encode(df):
    df = df.copy()

    print("\nApplying One-Hot Encoding...")
    print("-" * 60)

    available_columns = [
        column
        for column in ONE_HOT_COLUMNS
        if column in df.columns
    ]

    if not available_columns:

        print(
            "  No configured one-hot columns found."
        )

        return df

    # Handle missing values
    for column in available_columns:

        df[column] = df[column].fillna(
            "unknown"
        )

    df = pd.get_dummies(
        df,
        columns=available_columns,
        prefix=available_columns,
        drop_first=False,
        dtype=int,
    )

    for column in available_columns:

        print(
            f"  One-hot encoded: {column}"
        )

    return df

def apply_log1p(df):
    df = df.copy()

    print("\nApplying log1p Transformation...")
    print("-" * 60)

    for column in LOG1P_COLUMNS:

        if column not in df.columns:

            print(
                f"  Skipped: {column} "
                f"(column not found)"
            )

            continue
            
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        # Check for negative values
        if (
            df[column]
            .dropna()
            .lt(0)
            .any()
        ):

            raise ValueError(
                f"Negative values found in "
                f"'{column}'. "
                f"Cannot apply log1p transformation."
            )

        df[column] = np.log1p(
            df[column]
        )

        print(
            f"  log1p applied: {column}"
        )

    return df

# VALIDATE OUTPUT

def validate_output(
   df,
   original_df,
):
   print("\nValidating transformed dataset...")
   print("-" * 60)

   if "customer_unique_id" not in df.columns:
       raise ValueError("customer_unique_id is missing.")

   print("  Customer ID check: OK")

   if df["customer_unique_id"].duplicated().any():
       raise ValueError("Duplicate customer_unique_id values found.")

   print("  Duplicate customer ID check: OK")

   if len(df) != len(original_df):
       raise ValueError("Row count changed during transformation.")

   print("  Row count check: OK")

   if "churn_label" not in df.columns:
       raise ValueError("churn_label is missing.")

   print("  Churn label: OK")

   if "censored" not in df.columns:
       raise ValueError("censored is missing.")

   print("  Censored flag: OK")

   numeric_df = df.select_dtypes(include=np.number)
   for column in numeric_df.columns:
       values = pd.to_numeric(numeric_df[column], errors="coerce")
       if np.isinf(values).any():
           raise ValueError(f"Infinite values found in column: {column}")

   print("  Infinite value check: OK")

   for column in FREQUENCY_ENCODE_COLUMNS:
       if column in df.columns:
           raise ValueError(
               f"Original frequency encoded column still exists: {column}"
           )

       encoded_column = f"{column}_frequency"
       if encoded_column not in df.columns:
           raise ValueError(
               f"Frequency encoded column missing: {encoded_column}"
           )

   print("  Frequency encoding check: OK")

   for column in ONE_HOT_COLUMNS:
       if column in df.columns:
           raise ValueError(
               f"Original one-hot encoded column still exists: {column}"
           )

   print("  One-hot encoding check: OK")

   for column in LOG1P_COLUMNS:
       if column not in df.columns:
           continue

       values = pd.to_numeric(df[column], errors="coerce")
       if values.notna().any() and values.dropna().lt(0).any():
           raise ValueError(f"Negative value found after log1p in: {column}")

   print("  log1p transformation check: OK")

   important_columns = [
       "customer_unique_id",
       "churn_label",
       "censored",
       "first_purchase_date",
       "last_purchase_date",
       "reference_date",
   ]

   missing_columns = [
       column for column in important_columns if column not in df.columns
   ]

   if missing_columns:
       raise ValueError(
           "Important columns missing: " + ", ".join(missing_columns)
       )

   print("  ID/date/label preservation check: OK")
   print("\nOutput validation passed.")
   print(f"Output rows   : {len(df):,}")
   print(f"Output columns: {len(df.columns):,}")


def transform_data(df):
   """Apply the required feature encoding and transformation steps."""

   print("\n")
   print("=" * 70)
   print("STARTING FEATURE ENCODING AND TRANSFORMATION")
   print("=" * 70)

   df = frequency_encode(df)
   df = one_hot_encode(df)
   df = apply_log1p(df)

   return df

def save_to_database(df):
    print("\nSaving transformed features to MySQL...")
    print("-" * 60)

    try:

        df.to_sql(
            DATABASE_TABLE,
            con=engine,
            if_exists="replace",
            index=False,
            chunksize=5000,
        )

        print(
            f"Database table "
            f"'{DATABASE_TABLE}' "
            f"created/replaced successfully."
        )

        print(
            f"Rows inserted: {len(df):,}"
        )

        print(
            f"Columns inserted: {len(df.columns):,}"
        )

    except Exception as error:

        print(
            "\nERROR: Failed to save "
            "features_encoded to MySQL."
        )

        raise error

def main():

    print("\n")
    print("=" * 70)

    print(
        "CUSTOMER FEATURE ENCODING "
        "AND TRANSFORMATION"
    )

    print("=" * 70)

    original_df = load_data()

    transformed_df = transform_data(
        original_df
    )

    validate_output(
        transformed_df,
        original_df
    )

    save_to_database(
        transformed_df
    )

    print("\n")
    print("=" * 70)

    print(
        "PIPELINE COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)

    print(
        f"\nInput MySQL table  : "
        f"{INPUT_DATABASE_TABLE}"
    )

    print(
        f"Output MySQL table : "
        f"{DATABASE_TABLE}"
    )

    print(
        f"Input rows         : "
        f"{len(original_df):,}"
    )

    print(
        f"Output rows        : "
        f"{len(transformed_df):,}"
    )

if __name__ == "__main__":
    main()
