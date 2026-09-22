

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


TABLE_NAME = "customer_cohort_analysis"
INPUT_COLUMNS = [
    "customer_unique_id",
    "first_purchase_date",
    "order_purchase_timestamp",
    "payment_value",
    "total_orders",
]
OUTPUT_COLUMNS = [
    "cohort",
    "relative_month",
    "retention_rate (%)",
    "repeat_purchase_rate (%)",
    "average_clv",
    "generated_date",
]
CSV_FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
}


def _customer_intelligence_repository() -> Any:
    """Load the companion repository only when its functions are requested."""
    from app.segmentation.customer_analytics import customer_intelligence_repository

    return customer_intelligence_repository


def get_churn_predictions() -> pd.DataFrame:
    """Return database-backed churn predictions from the companion repository."""
    return _customer_intelligence_repository().get_churn_predictions()


def build_customer_intelligence_base() -> pd.DataFrame:
    """Return the validated feature/prediction merge for segmentation."""
    return _customer_intelligence_repository().build_customer_intelligence_base()


def validate_merged_dataset(merged: pd.DataFrame) -> None:
    """Validate a customer-intelligence merge before downstream segmentation."""
    _customer_intelligence_repository().validate_merged_dataset(merged)


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")


def build_scoped_input(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    payments: pd.DataFrame,
) -> pd.DataFrame:
    """Merge raw tables at one row per customer-order.

    Payment rows are summed per order because Olist permits multiple payment
    records for one order.  Orders are never collapsed across a customer.
    """
    _require_columns(
        customers, {"customer_id", "customer_unique_id"}, "customers"
    )
    _require_columns(
        orders, {"order_id", "customer_id", "order_purchase_timestamp"}, "orders"
    )
    _require_columns(payments, {"order_id", "payment_value"}, "order_payments")

    customer_map = customers[["customer_id", "customer_unique_id"]].dropna()
    order_frame = orders[
        ["order_id", "customer_id", "order_purchase_timestamp"]
    ].copy()
    order_frame["order_purchase_timestamp"] = pd.to_datetime(
        order_frame["order_purchase_timestamp"], errors="coerce"
    )
    order_frame = order_frame.dropna(
        subset=["order_id", "customer_id", "order_purchase_timestamp"]
    ).drop_duplicates("order_id")

    payment_totals = (
        payments[["order_id", "payment_value"]]
        .assign(payment_value=lambda frame: pd.to_numeric(
            frame["payment_value"], errors="coerce"
        ).fillna(0.0))
        .groupby("order_id", as_index=False, dropna=False)["payment_value"]
        .sum()
    )
    merged = order_frame.merge(customer_map, on="customer_id", how="inner")
    merged = merged.merge(payment_totals, on="order_id", how="left")
    merged["payment_value"] = merged["payment_value"].fillna(0.0)
    merged = merged.dropna(subset=["customer_unique_id"])

    first_dates = merged.groupby("customer_unique_id")[
        "order_purchase_timestamp"
    ].transform("min")
    merged["first_purchase_date"] = first_dates
    merged["total_orders"] = merged.groupby("customer_unique_id")[
        "order_id"
    ].transform("nunique").astype("int64")

    result = merged[
        [
            "customer_unique_id",
            "first_purchase_date",
            "order_purchase_timestamp",
            "payment_value",
            "total_orders",
        ]
    ].sort_values(["customer_unique_id", "order_purchase_timestamp"])
    return result.reset_index(drop=True)


def load_scoped_input(
    data_dir: str | Path | None = None,
    connection: Any | None = None,
) -> pd.DataFrame:
    """Load raw source tables from the database, with CSV support for fixtures."""
    if connection is None and data_dir is None:
        from app.Database.database import engine

        connection = engine

    if connection is not None:
        customers = pd.read_sql(
            "SELECT customer_id, customer_unique_id FROM customers", connection
        )
        orders = pd.read_sql(
            "SELECT order_id, customer_id, order_purchase_timestamp FROM orders",
            connection,
        )
        payments = pd.read_sql(
            "SELECT order_id, payment_value FROM order_payments", connection
        )
    else:
        if data_dir is None:
            raise ValueError("Provide data_dir or a database connection.")
        root = Path(data_dir)
        paths = {name: root / filename for name, filename in CSV_FILES.items()}
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing cohort source file(s): " + ", ".join(missing)
            )
        customers = pd.read_csv(paths["customers"])
        orders = pd.read_csv(paths["orders"])
        payments = pd.read_csv(paths["payments"])
    return build_scoped_input(customers, orders, payments)


def calculate_cohort_analysis(
    scoped_input: pd.DataFrame,
    *,
    min_orders: int = 1,
    generated_date: date | str | None = None,
) -> pd.DataFrame:
    """Calculate retention, repeat purchase rate, and average cohort CLV.

    ``min_orders`` is deliberately local to this module.  The default keeps
    every valid purchaser (including one-time customers), while callers can
    require repeat-capable customers without altering other model datasets.
    """
    if min_orders < 1:
        raise ValueError("min_orders must be at least 1")
    _require_columns(scoped_input, set(INPUT_COLUMNS), "scoped input")

    frame = scoped_input.copy()
    frame["first_purchase_date"] = pd.to_datetime(
        frame["first_purchase_date"], errors="coerce"
    )
    frame["order_purchase_timestamp"] = pd.to_datetime(
        frame["order_purchase_timestamp"], errors="coerce"
    )
    frame["payment_value"] = pd.to_numeric(
        frame["payment_value"], errors="coerce"
    ).fillna(0.0)
    frame = frame.dropna(
        subset=[
            "customer_unique_id",
            "first_purchase_date",
            "order_purchase_timestamp",
        ]
    )
    frame = frame[frame["total_orders"].fillna(0).astype(int) >= min_orders]
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    frame["cohort_month"] = frame["first_purchase_date"].dt.to_period("M")
    frame["activity_month"] = frame["order_purchase_timestamp"].dt.to_period("M")
    frame["relative_month"] = (
        (frame["activity_month"].dt.year - frame["cohort_month"].dt.year) * 12
        + frame["activity_month"].dt.month
        - frame["cohort_month"].dt.month
    )
    frame = frame[frame["relative_month"] >= 0]
    cohort_sizes = frame.groupby("cohort_month")["customer_unique_id"].nunique()
    customer_summary = frame.groupby("customer_unique_id").agg(
        cohort_month=("cohort_month", "first"),
        purchase_count=("order_purchase_timestamp", "size"),
    )
    repeaters = customer_summary.assign(
        is_repeater=customer_summary["purchase_count"].ge(2)
    ).groupby("cohort_month")["is_repeater"].sum()
    max_month = frame["activity_month"].max()
    rows: list[dict[str, object]] = []
    for cohort_month, cohort_frame in frame.groupby("cohort_month", sort=True):
        last_relative_month = (max_month.year - cohort_month.year) * 12 + (
            max_month.month - cohort_month.month
        )
        cohort_size = int(cohort_sizes[cohort_month])
        average_clv = cohort_frame["payment_value"].sum() / cohort_size
        active = cohort_frame.groupby("relative_month")[
            "customer_unique_id"
        ].nunique()
        repeated_count = int(repeaters.get(cohort_month, 0))
        for relative_month in range(last_relative_month + 1):
            rows.append(
                {
                    "cohort": cohort_month.strftime("%b %Y"),
                    "relative_month": f"M{relative_month}",
                    "retention_rate (%)": round(
                        float(active.get(relative_month, 0)) / cohort_size * 100, 2
                    ),
                    "repeat_purchase_rate (%)": round(
                        repeated_count / cohort_size * 100, 2
                    ),
                    "average_clv": round(average_clv, 2),
                    "generated_date": (
                        pd.Timestamp(generated_date).date()
                        if generated_date is not None
                        else date.today()
                    ),
                }
            )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def write_cohort_analysis(
    result: pd.DataFrame,
    connection: Any,
    *,
    if_exists: str = "replace",
) -> None:
    """Write the final table without touching any other analytics tables."""
    result.to_sql(TABLE_NAME, con=connection, if_exists=if_exists, index=False)


def run(
    *,
    data_dir: str | Path | None = None,
    connection: Any | None = None,
    output_connection: Any | None = None,
    min_orders: int = 1,
    generated_date: date | str | None = None,
) -> pd.DataFrame:
    """Build the database-backed extract, calculate cohorts, and persist it."""
    if connection is None and data_dir is None:
        from app.Database.database import engine

        connection = engine
    scoped = load_scoped_input(data_dir=data_dir, connection=connection)
    result = calculate_cohort_analysis(
        scoped, min_orders=min_orders, generated_date=generated_date
    )
    if output_connection is None and connection is not None:
        output_connection = connection
    if output_connection is not None:
        write_cohort_analysis(result, output_connection)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Optional CSV fixture directory; database input is the default.",
    )
    parser.add_argument(
        "--database-url",
        help="Optional SQLAlchemy URL overriding the configured project database.",
    )
    parser.add_argument("--min-orders", type=int, default=1)
    args = parser.parse_args()
    connection = None
    if args.database_url:
        from sqlalchemy import create_engine

        connection = create_engine(args.database_url)
    result = run(
        data_dir=args.data_dir,
        connection=connection,
        min_orders=args.min_orders,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
