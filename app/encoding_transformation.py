
from pathlib import Path

import numpy as np
import pandas as pd

from app.database import engine


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Optional CSV backup output
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "features_encoded.csv"
)


# ============================================================
# MYSQL TABLE NAMES
# ============================================================

# INPUT TABLE
# This table should contain the data that was previously
# stored in customer_features_with_labels.csv
INPUT_DATABASE_TABLE = "customer_features_with_labels"

# OUTPUT TABLE
DATABASE_TABLE = "features_encoded"


# ============================================================
# ONE-HOT ENCODING
# ============================================================

# Low-cardinality categorical features
ONE_HOT_COLUMNS = [
    "latest_order_status",
    "preferred_payment_type",
]


# ============================================================
# FREQUENCY ENCODING
# ============================================================

# High-cardinality categorical features
FREQUENCY_ENCODE_COLUMNS = [
    "dominant_product_category",
    "customer_city_state",
]


# ============================================================
# LOG1P TRANSFORMATION
# ============================================================

# Selected skewed numerical/count features
#
# log1p(x) = log(1 + x)

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


# ============================================================
# LOAD DATA FROM MYSQL
# ============================================================

def load_data():
    """
    Load the customer-level dataset directly from MySQL.

    The input table replaces the CSV file that was previously
    used by the pipeline.
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


# ============================================================
# FREQUENCY ENCODING
# ============================================================

def frequency_encode(df):
    """
    Apply frequency encoding to high-cardinality
    categorical features.

    Each category is replaced by its proportion
    in the dataset.
    """

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

        # Calculate category frequency
        frequency_map = (
            df[column]
            .value_counts(normalize=True)
        )

        # New encoded column
        encoded_column = (
            f"{column}_frequency"
        )

        # Apply frequency mapping
        df[encoded_column] = (
            df[column]
            .map(frequency_map)
            .fillna(0)
        )

        # Remove original categorical column
        df.drop(
            columns=[column],
            inplace=True
        )

        print(
            f"  {column} -> {encoded_column}"
        )

    return df


# ============================================================
# ONE-HOT ENCODING
# ============================================================

def one_hot_encode(df):
    """
    Apply one-hot encoding to low-cardinality
    categorical features.
    """

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

    # Apply one-hot encoding
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


# ============================================================
# LOG1P TRANSFORMATION
# ============================================================

def apply_log1p(df):
    """
    Apply log1p transformation to selected
    skewed numerical/count features.

    log1p(x) = log(1 + x)
    """

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

        # Convert column to numeric
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

        # Apply log1p
        df[column] = np.log1p(
            df[column]
        )

        print(
            f"  log1p applied: {column}"
        )

    return df


# ============================================================
# VALIDATE OUTPUT
# ============================================================

def validate_output(
    df,
    original_df
):
    """
    Validate the final encoded and transformed dataset.
    """

    print("\nValidating transformed dataset...")
    print("-" * 60)

    # --------------------------------------------------------
    # Customer ID
    # --------------------------------------------------------

    if "customer_unique_id" not in df.columns:

        raise ValueError(
            "customer_unique_id is missing."
        )

    print(
        "  Customer ID check: OK"
    )


    # --------------------------------------------------------
    # Duplicate Customer ID
    # --------------------------------------------------------

    if (
        df["customer_unique_id"]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate customer_unique_id "
            "values found."
        )

    print(
        "  Duplicate customer ID check: OK"
    )


    # --------------------------------------------------------
    # Row Count
    # --------------------------------------------------------

    if len(df) != len(original_df):

        raise ValueError(
            "Row count changed during transformation."
        )

    print(
        "  Row count check: OK"
    )


    # --------------------------------------------------------
    # Churn Label
    # --------------------------------------------------------

    if "churn_label" not in df.columns:

        raise ValueError(
            "churn_label is missing."
        )

    print(
        "  Churn label: OK"
    )


    # --------------------------------------------------------
    # Censored Flag
    # --------------------------------------------------------

    if "censored" not in df.columns:

        raise ValueError(
            "censored is missing."
        )

    print(
        "  Censored flag: OK"
    )


    # --------------------------------------------------------
    # Infinite Values
    # --------------------------------------------------------

    numeric_df = df.select_dtypes(
        include=np.number
    )

    for column in numeric_df.columns:

        values = pd.to_numeric(
            numeric_df[column],
            errors="coerce"
        )

        if np.isinf(values).any():

            raise ValueError(
                f"Infinite values found "
                f"in column: {column}"
            )

    print(
        "  Infinite value check: OK"
    )


    # --------------------------------------------------------
    # Frequency Encoding Check
    # --------------------------------------------------------

    for column in FREQUENCY_ENCODE_COLUMNS:

        # Original column should be removed
        if column in df.columns:

            raise ValueError(
                f"Original frequency encoded "
                f"column still exists: {column}"
            )

        # Encoded column should exist
        encoded_column = (
            f"{column}_frequency"
        )

        if encoded_column not in df.columns:

            raise ValueError(
                f"Frequency encoded column "
                f"missing: {encoded_column}"
            )

    print(
        "  Frequency encoding check: OK"
    )


    # --------------------------------------------------------
    # One-Hot Encoding Check
    # --------------------------------------------------------

    for column in ONE_HOT_COLUMNS:

        # Original categorical column
        # should no longer exist

        if column in df.columns:

            raise ValueError(
                f"Original one-hot encoded "
                f"column still exists: {column}"
            )

    print(
        "  One-hot encoding check: OK"
    )


    # --------------------------------------------------------
    # Log1p Check
    # --------------------------------------------------------

    for column in LOG1P_COLUMNS:

        if column not in df.columns:

            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if values.notna().any():

            if (
                values
                .dropna()
                .lt(0)
                .any()
            ):

                raise ValueError(
                    f"Negative value found "
                    f"after log1p in: {column}"
                )

    print(
        "  log1p transformation check: OK"
    )


    # --------------------------------------------------------
    # Important Columns
    # --------------------------------------------------------

    important_columns = [
        "customer_unique_id",
        "churn_label",
        "censored",
        "first_purchase_date",
        "last_purchase_date",
        "reference_date",
    ]

    missing_columns = [
        column
        for column in important_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Important columns missing: "
            + ", ".join(
                missing_columns
            )
        )

    print(
        "  ID/date/label preservation check: OK"
    )


    # --------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------

    print("\nOutput validation passed.")

    print(
        f"Output rows   : {len(df):,}"
    )

    print(
        f"Output columns: {len(df.columns):,}"
    )


# ============================================================
# TRANSFORMATION PIPELINE
# ============================================================

def transform_data(df):
    """
    Apply all required transformations:

    1. Frequency Encoding
    2. One-Hot Encoding
    3. log1p Transformation
    """

    print("\n")
    print("=" * 70)

    print(
        "STARTING FEATURE ENCODING "
        "AND TRANSFORMATION"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Step 1: Frequency Encoding
    # --------------------------------------------------------

    df = frequency_encode(df)


    # --------------------------------------------------------
    # Step 2: One-Hot Encoding
    # --------------------------------------------------------

    df = one_hot_encode(df)


    # --------------------------------------------------------
    # Step 3: log1p Transformation
    # --------------------------------------------------------

    df = apply_log1p(df)


    return df


# ============================================================
# SAVE OUTPUT TO CSV
# ============================================================

def save_output(df):
    """
    Save the final transformed dataset to CSV
    as an optional backup.
    """

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n")
    print("=" * 70)

    print(
        "CSV BACKUP CREATED"
    )

    print("=" * 70)

    print(
        f"\nBackup file:"
        f"\n{OUTPUT_PATH}"
    )

    print(
        f"\nFinal shape: {df.shape}"
    )


# ============================================================
# SAVE OUTPUT TO MYSQL
# ============================================================

def save_to_database(df):
    """
    Save the transformed features to MySQL.
    """

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


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)

    print(
        "CUSTOMER FEATURE ENCODING "
        "AND TRANSFORMATION"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Step 1: Load input dataset FROM MYSQL
    # --------------------------------------------------------

    original_df = load_data()


    # --------------------------------------------------------
    # Step 2: Apply transformations
    # --------------------------------------------------------

    transformed_df = transform_data(
        original_df
    )


    # --------------------------------------------------------
    # Step 3: Validate final dataset
    # --------------------------------------------------------

    validate_output(
        transformed_df,
        original_df
    )


    # --------------------------------------------------------
    # Step 4: Save optional CSV backup
    # --------------------------------------------------------

    save_output(
        transformed_df
    )


    # --------------------------------------------------------
    # Step 5: Save transformed data to MySQL
    # --------------------------------------------------------

    save_to_database(
        transformed_df
    )


    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

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



# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
