import sys
from pathlib import Path
import pandas as pd
import yaml
from sqlalchemy import text

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent

for candidate in (str(PROJECT_ROOT), str(APP_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from app.Database.database import engine
except ModuleNotFoundError:
    from Database.database import engine


# Configuration for churn labeling and output paths.
CONFIG_PATH = APP_DIR / "config" / "label_config.yaml"


def load_config() -> dict:
    """Load churn-label config."""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found at: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if "reference_date" not in config:
        raise ValueError(
            "label_config.yaml must define 'reference_date'."
        )

    config["reference_date"] = pd.Timestamp(
        config["reference_date"]
    )

    return config


# Fetch valid orders up to the reference date.
def fetch_valid_orders(
    db_engine,
    reference_date: pd.Timestamp,
    excluded_statuses: list,
) -> pd.DataFrame:
    """Return valid orders up to the reference date."""

    query = text(
        """
        SELECT
            c.customer_unique_id,
            o.order_id,
            o.order_status,
            o.order_purchase_timestamp
        FROM orders o
        INNER JOIN customers c
            ON o.customer_id = c.customer_id
        WHERE o.order_status NOT IN :excluded_statuses
          AND o.order_purchase_timestamp IS NOT NULL
          AND o.order_purchase_timestamp <= :reference_date
        """
    )

    with db_engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={
                "reference_date": reference_date,
                "excluded_statuses": (
                    tuple(excluded_statuses)
                    if excluded_statuses
                    else ("__NO_EXCLUDED_STATUS__",)
                ),
            },
        )

    if not df.empty:
        df["order_purchase_timestamp"] = pd.to_datetime(
            df["order_purchase_timestamp"]
        )

    return df


# Build customer churn labels from order history.
def build_labels():
    """Build customer churn labels from order history."""

    print("\nLoading configuration and database engine...")

    config = load_config()

    reference_date = config["reference_date"]

    return_window_days = int(
        config.get("return_window_days", 180)
    )

    exclude_censored = bool(
        config.get("exclude_censored_customers", False)
    )

    # Keep excluded statuses empty so canceled orders still contribute to recency.
    excluded_statuses = []

    print(f"Configuration File:      {CONFIG_PATH}")
    print(f"Reference Date:          {reference_date.date()}")
    print(f"Return Window Days:      {return_window_days}")
    print("Excluded Order Statuses: []")
    print(
        f"Exclude Censored:        {exclude_censored}"
    )

    # Fetch orders up to the reference date.
    orders = fetch_valid_orders(
        engine,
        reference_date,
        excluded_statuses,
    )

    if orders.empty:
        print("\nNo valid orders found.")

        empty_result = pd.DataFrame(
            columns=[
                "customer_unique_id",
                "first_order_date",
                "last_order_date",
                "days_since_last_order",
                "num_valid_orders",
                "censored",
                "label",
            ]
        )

        return empty_result, reference_date

    # Sort orders to support time-based customer-level analysis.
    orders = orders.sort_values(
        [
            "customer_unique_id",
            "order_purchase_timestamp",
        ]
    ).reset_index(drop=True)

    # Aggregate order activity at the customer level.
    grouped = orders.groupby(
        "customer_unique_id"
    )

    result = (
        grouped.agg(
            first_order_date=(
                "order_purchase_timestamp",
                "min",
            ),
            last_order_date=(
                "order_purchase_timestamp",
                "max",
            ),
            num_valid_orders=(
                "order_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    # Measure recency from the most recent valid order.
    result["days_since_last_order"] = (
        reference_date - result["last_order_date"]
    ).dt.total_seconds() / 86400.0

    # Compute the gap between consecutive orders for repeat-purchase detection.
    orders["prev_order_date"] = (
        orders
        .groupby("customer_unique_id")[
            "order_purchase_timestamp"
        ]
        .shift(1)
    )

    orders["gap_to_prev"] = (
        orders["order_purchase_timestamp"]
        - orders["prev_order_date"]
    ).dt.total_seconds() / 86400.0

    # Identify customers who returned within the configured return window.
    repeat_customers = set(
        orders.loc[
            orders["gap_to_prev"] <= return_window_days,
            "customer_unique_id",
        ]
    )

    result["has_repeat"] = (
        result["customer_unique_id"].isin(
            repeat_customers
        )
    )

    # Mark censored customers whose observation window has not yet fully elapsed.
    result["censored"] = (
        (~result["has_repeat"])
        & (
            result["days_since_last_order"]
            < return_window_days
        )
    )

    # Assign final churn labels: retained, churned, or censored.
    result["label"] = pd.NA

    # Mark customers who returned within the retention window as retained.
    result.loc[
        result["has_repeat"],
        "label",
    ] = 0

    # Mark customers who exceeded the observation window without returning as churned.
    result.loc[
        (~result["has_repeat"])
        & (
            result["days_since_last_order"]
            >= return_window_days
        ),
        "label",
    ] = 1

    # Drop the temporary recurrence flag after labeling is complete.
    result = result.drop(
        columns=["has_repeat"]
    )

    # Drop censored customers when the configuration requests a fully observed dataset.
    if exclude_censored:
        result = result[
            ~result["censored"]
        ].reset_index(drop=True)

    # Preserve a consistent output schema for downstream analysis and storage.
    columns = [
        "customer_unique_id",
        "first_order_date",
        "last_order_date",
        "days_since_last_order",
        "num_valid_orders",
        "censored",
        "label",
    ]

    result = result[columns]

    # Use pandas nullable integers so censored customers remain representable in the output.
    result["label"] = result["label"].astype("Int64")

    return result, reference_date


# Persist churn labels to MySQL in a production-friendly format.
def save_labels_to_db(
    db_engine,
    df: pd.DataFrame,
    table_name: str = "customer_churn_labels",
):
    """Persist churn labels to MySQL."""

    print(
        f"\nSaving labels to MySQL table `{table_name}`..."
    )

    db_df = df.copy()

    # Convert missing labels to SQL NULL values before writing to MySQL.
    db_df["label"] = (
        db_df["label"]
        .astype(object)
        .where(
            db_df["label"].notna(),
            None,
        )
    )

    with db_engine.begin() as connection:
        db_df.to_sql(
            table_name,
            con=connection,
            if_exists="replace",
            index=False,
            chunksize=5000,
        )

    print(
        f"Saved -> MySQL table `{table_name}` "
        f"({len(db_df):,} rows)"
    )


# Entry point for generating and persisting churn labels.
if __name__ == "__main__":

    labels, reference_date = build_labels()

    if labels.empty:
        print("\nNo labels generated.")
        print("Process completed.")
        raise SystemExit(0)

    # Summarize the final churn label distribution.
    total_customers = len(labels)

    retained = (
        labels["label"] == 0
    ).sum()

    churned = (
        labels["label"] == 1
    ).sum()

    censored = (
        labels["censored"]
    ).sum()

    print("\n" + "=" * 65)
    print("HYBRID CUSTOMER CHURN LABEL BREAKDOWN")
    print("=" * 65)

    print(
        f"Reference Date:          {reference_date.date()}"
    )

    print(
        f"Total Unique Customers:  {total_customers:,}"
    )

    print(
        f"Retained (Label 0):      "
        f"{retained:,} "
        f"({retained / total_customers * 100:.2f}%)"
    )

    print(
        f"Churned (Label 1):       "
        f"{churned:,} "
        f"({churned / total_customers * 100:.2f}%)"
    )

    print(
        f"Censored Customers:      "
        f"{censored:,} "
        f"({censored / total_customers * 100:.2f}%)"
    )

    print("=" * 65)

    # Save the final label set to MySQL.
    save_labels_to_db(
        engine,
        labels,
        table_name="customer_churn_labels",
    )

    print("\nProcess completed successfully.")
