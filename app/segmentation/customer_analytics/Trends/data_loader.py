"""Load and normalize data used by the historical trend pipeline."""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import yaml
from sqlalchemy import bindparam, text


def _find_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "app").exists() and (parent / "data").exists():
            return parent
    return Path(__file__).resolve().parents[4]


ROOT = _find_root()
APP_DIR = ROOT / "app"
DATA_DIR = ROOT / "data"
CONFIG_PATH = APP_DIR / "config" / "trend_config.yaml"


class TrendDataLoader:
    """
    Loads and normalizes transactions, order values, and customer churn records
    for time-series historical trend analysis.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        db_engine=None,
    ):
        self.config_path = config_path or CONFIG_PATH
        self.data_dir = data_dir or DATA_DIR
        self.config = self._load_config()
        self.db_engine = db_engine or self._init_db_engine()

    def _load_config(self) -> dict:
        """Load trend configuration."""
        if self.config_path.exists():
            with self.config_path.open(encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {
            "reference_date": "2018-10-17",
            "return_window_days": 180,
            "excluded_order_statuses": ["canceled", "unavailable"],
        }

    def _init_db_engine(self):
        """Try loading the project SQLAlchemy database engine."""
        try:
            from app.Database.database import engine
            return engine
        except Exception:
            try:
                from Database.database import engine
                return engine
            except Exception:
                return None

    def load_transaction_data(
        self,
        reference_date: Optional[pd.Timestamp] = None,
        excluded_statuses: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Load order-level transaction data with revenue and customer mapping.
        Returns a DataFrame with:
        ['order_id', 'customer_unique_id', 'order_purchase_timestamp', 'order_status', 'order_value']
        """
        ref_date = reference_date or pd.Timestamp(self.config.get("reference_date", "2018-10-17"))
        statuses = (
            excluded_statuses
            if excluded_statuses is not None
            else self.config.get("excluded_order_statuses", ["canceled", "unavailable"])
        )

        # 1. Attempt loading from MySQL database
        if self.db_engine is not None:
            try:
                df = self._load_from_database(ref_date, statuses)
                if not df.empty:
                    return df
            except Exception as e:
                print(f"[TrendDataLoader] DB load failed ({e}), falling back to CSV files.")

        # 2. Fallback: Load and join directly from CSV files
        return self._load_from_csv(ref_date, statuses)

    def _load_from_database(
        self,
        reference_date: pd.Timestamp,
        excluded_statuses: list,
    ) -> pd.DataFrame:
        """Query orders joined with customers and payments from MySQL using expanding bind."""
        statuses = list(excluded_statuses) if excluded_statuses else ["__NONE__"]

        query = text("""
            SELECT
                o.order_id,
                c.customer_unique_id,
                o.order_purchase_timestamp,
                o.order_status,
                COALESCE(p.total_payment, i.total_items, 0.0) AS order_value
            FROM orders o
            INNER JOIN customers c
                ON o.customer_id = c.customer_id
            LEFT JOIN (
                SELECT order_id, SUM(payment_value) AS total_payment
                FROM order_payments
                GROUP BY order_id
            ) p ON o.order_id = p.order_id
            LEFT JOIN (
                SELECT order_id, SUM(COALESCE(price, 0) + COALESCE(freight_value, 0)) AS total_items
                FROM order_items
                GROUP BY order_id
            ) i ON o.order_id = i.order_id
            WHERE o.order_purchase_timestamp IS NOT NULL
              AND o.order_purchase_timestamp <= :reference_date
              AND o.order_status NOT IN :excluded_statuses
        """).bindparams(
            bindparam("reference_date"),
            bindparam("excluded_statuses", expanding=True),
        )

        with self.db_engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={
                    "reference_date": reference_date,
                    "excluded_statuses": statuses,
                },
            )

        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
        df["order_value"] = pd.to_numeric(df["order_value"], errors="coerce").fillna(0.0)
        return df

    def _load_from_csv(
        self,
        reference_date: pd.Timestamp,
        excluded_statuses: list,
    ) -> pd.DataFrame:
        """Load and merge orders, customers, and payments from raw CSV files."""
        orders_file = self.data_dir / "olist_orders_dataset.csv"
        customers_file = self.data_dir / "olist_customers_dataset.csv"
        payments_file = self.data_dir / "olist_order_payments_dataset.csv"
        items_file = self.data_dir / "olist_order_items_dataset.csv"

        if not orders_file.exists() or not customers_file.exists():
            raise FileNotFoundError(
                f"Source datasets not found in {self.data_dir}. "
                "Ensure CSV files are present."
            )

        orders = pd.read_csv(orders_file)
        customers = pd.read_csv(customers_file)

        # Parse timestamps and filter
        orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
        orders = orders[
            orders["order_purchase_timestamp"].notna()
            & (orders["order_purchase_timestamp"] <= reference_date)
        ]

        if excluded_statuses:
            orders = orders[~orders["order_status"].isin(excluded_statuses)]

        # Merge with customers for customer_unique_id
        merged = orders.merge(
            customers[["customer_id", "customer_unique_id"]],
            on="customer_id",
            how="inner",
        )

        # Calculate revenue per order from payments (or items fallback)
        order_values = None
        if payments_file.exists():
            payments = pd.read_csv(payments_file)
            order_values = payments.groupby("order_id")["payment_value"].sum().reset_index()
            order_values.rename(columns={"payment_value": "order_value"}, inplace=True)
        elif items_file.exists():
            items = pd.read_csv(items_file)
            items["item_total"] = items["price"].fillna(0) + items["freight_value"].fillna(0)
            order_values = items.groupby("order_id")["item_total"].sum().reset_index()
            order_values.rename(columns={"item_total": "order_value"}, inplace=True)

        if order_values is not None:
            merged = merged.merge(order_values, on="order_id", how="left")
            merged["order_value"] = merged["order_value"].fillna(0.0)
        else:
            merged["order_value"] = 0.0

        return merged[
            [
                "order_id",
                "customer_unique_id",
                "order_purchase_timestamp",
                "order_status",
                "order_value",
            ]
        ].copy()

    def load_customer_churn_data(
        self,
        reference_date: Optional[pd.Timestamp] = None,
        return_window_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load customer churn labels and order timelines.
        Returns a DataFrame with:
        ['customer_unique_id', 'first_order_date', 'last_order_date', 'days_since_last_order', 'num_valid_orders', 'censored', 'label']
        """
        ref_date = reference_date or pd.Timestamp(self.config.get("reference_date", "2018-10-17"))
        return_window = int(return_window_days or self.config.get("return_window_days", 180))

        # Try database table first
        if self.db_engine is not None:
            try:
                with self.db_engine.connect() as conn:
                    query = text("SELECT * FROM customer_churn_labels")
                    df = pd.read_sql(query, conn)
                    if not df.empty and "label" in df.columns and "censored" in df.columns:
                        df["first_order_date"] = pd.to_datetime(df["first_order_date"])
                        df["last_order_date"] = pd.to_datetime(df["last_order_date"])
                        return df
            except Exception:
                pass

        # Fallback: compute dynamically using transaction data
        orders = self.load_transaction_data(reference_date=ref_date)
        if orders.empty:
            return pd.DataFrame(
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

        orders = orders.sort_values(["customer_unique_id", "order_purchase_timestamp"]).reset_index(drop=True)
        grouped = orders.groupby("customer_unique_id")

        result = grouped.agg(
            first_order_date=("order_purchase_timestamp", "min"),
            last_order_date=("order_purchase_timestamp", "max"),
            num_valid_orders=("order_id", "nunique"),
        ).reset_index()

        result["days_since_last_order"] = (ref_date - result["last_order_date"]).dt.total_seconds() / 86400.0

        orders["prev_order_date"] = orders.groupby("customer_unique_id")["order_purchase_timestamp"].shift(1)
        orders["gap_to_prev"] = (orders["order_purchase_timestamp"] - orders["prev_order_date"]).dt.total_seconds() / 86400.0

        repeat_customers = set(orders.loc[orders["gap_to_prev"] <= return_window, "customer_unique_id"])
        result["has_repeat"] = result["customer_unique_id"].isin(repeat_customers)
        result["censored"] = (~result["has_repeat"]) & (result["days_since_last_order"] < return_window)

        result["label"] = pd.NA
        result.loc[result["has_repeat"], "label"] = 0
        result.loc[(~result["has_repeat"]) & (result["days_since_last_order"] >= return_window), "label"] = 1
        result = result.drop(columns=["has_repeat"])
        result["label"] = result["label"].astype("Int64")

        return result
