from pathlib import Path
import os
from typing import Any
import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = next(
    (
        PROJECT_ROOT / filename
        for filename in ("label_config.yaml", "label_config (1).yaml")
        if (PROJECT_ROOT / filename).exists()
    ),
    PROJECT_ROOT / "label_config.yaml",
)


def _load_local_env(env_path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding environment variables."""
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_local_env(PROJECT_ROOT / ".env")

DB_CONFIG = {
    "host": os.environ["DB_HOST"],
    "port": int(os.environ["DB_PORT"]),
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "database": os.environ["DB_NAME"],
}

RAW_TABLES = (
    "customers",
    "orders",
    "order_items",
    "order_payments",
    "order_reviews",
    "products",
    "sellers",
    "geolocation",
)
PRIMARY_KEYS = {
    "customers": "customer_id",
    "sellers": "seller_id",
    "products": "product_id",
    "orders": "order_id",
}
ORDER_TIMESTAMP_COLUMN = "order_purchase_timestamp"


def _get_engine():
    """Create a SQLAlchemy engine for the configured MySQL database."""
    return create_engine(
        URL.create(
            drivername="mysql+pymysql",
            username=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            database=DB_CONFIG["database"],
        )
    )


def _report_cleaning(table_name: str, rule: str, before: int, after: int) -> None:
    dropped = before - after
    if dropped:
        print(f"[{table_name}] {rule}: dropped {dropped:,} row(s)")


def clean_dataframe(
    df: pd.DataFrame,
    table_name: str,
    valid_keys: dict[str, set] | None = None,
) -> pd.DataFrame:
    """Clean a DataFrame that has already been loaded from MySQL."""
    valid_keys = valid_keys or {}
    df = df.copy()

    for column in df.select_dtypes(include=["object", "string"]):
        df[column] = df[column].str.strip().replace("", pd.NA)

    def drop_duplicates(subset: list[str]) -> None:
        nonlocal df
        existing = [column for column in subset if column in df.columns]
        if not existing:
            return
        before = len(df)
        df = df.drop_duplicates(subset=existing)
        _report_cleaning(table_name, f"duplicate rows ({existing})", before, len(df))

    def require(columns: list[str]) -> None:
        nonlocal df
        existing = [column for column in columns if column in df.columns]
        if not existing:
            return
        before = len(df)
        df = df.dropna(subset=existing)
        _report_cleaning(table_name, f"missing required columns ({existing})", before, len(df))

    def numeric(columns: list[str]) -> None:
        for column in columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

    def non_negative(columns: list[str]) -> None:
        nonlocal df
        before = len(df)
        for column in columns:
            if column in df.columns:
                df = df[df[column].isna() | (df[column] >= 0)]
        _report_cleaning(table_name, f"negative values ({columns})", before, len(df))

    def in_range(column: str, low: float, high: float) -> None:
        if column in df.columns:
            invalid = (df[column] < low) | (df[column] > high)
            df.loc[invalid, column] = pd.NA

    def remove_orphans(column: str, parent: str) -> None:
        nonlocal df
        if parent not in valid_keys or column not in df.columns:
            return
        before = len(df)
        df = df[df[column].isin(valid_keys[parent])]
        _report_cleaning(table_name, f"orphaned {column}", before, len(df))

    if table_name == "customers":
        drop_duplicates(["customer_id"])
        require(["customer_id"])
    elif table_name == "sellers":
        drop_duplicates(["seller_id"])
        require(["seller_id"])
    elif table_name == "products":
        drop_duplicates(["product_id"])
        measurements = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
        numeric(measurements)
        require(["product_id"])
        non_negative(measurements)
    elif table_name == "orders":
        drop_duplicates(["order_id"])
        remove_orphans("customer_id", "customers")
        require(["order_id", "customer_id", "order_status", ORDER_TIMESTAMP_COLUMN])
    elif table_name == "order_items":
        numeric(["price", "freight_value"])
        require(["order_id", "product_id", "seller_id", "price", "freight_value"])
        non_negative(["price", "freight_value"])
        remove_orphans("order_id", "orders")
        remove_orphans("product_id", "products")
        remove_orphans("seller_id", "sellers")
    elif table_name == "order_payments":
        numeric(["payment_value"])
        require(["order_id", "payment_value"])
        non_negative(["payment_value"])
        remove_orphans("order_id", "orders")
    elif table_name == "order_reviews":
        numeric(["review_score"])
        in_range("review_score", 1, 5)
        require(["review_id", "order_id", "review_score", "review_creation_date"])
        remove_orphans("order_id", "orders")
        for column in ("review_comment_title", "review_comment_message"):
            if column in df.columns:
                df[column] = df[column].fillna("")
    elif table_name == "geolocation":
        numeric(["geolocation_lat", "geolocation_lng"])
        require(["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"])
        in_range("geolocation_lat", -90, 90)
        in_range("geolocation_lng", -180, 180)
        require(["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"])

    before = len(df)
    df = df[df.isna().mean(axis=1) <= 0.5]
    _report_cleaning(table_name, "more than 50% missing values", before, len(df))
    return df.reset_index(drop=True)


def ingest_table(
    table_name: str,
    connection,
    valid_keys: dict[str, set],
) -> pd.DataFrame:
    """Read and clean one table directly from the database."""
    if table_name not in RAW_TABLES:
        raise ValueError(f"Unknown database table: {table_name}")
    raw = pd.read_sql_query(text(f"SELECT * FROM {table_name}"), connection)
    cleaned = clean_dataframe(raw, table_name, valid_keys)
    primary_key = PRIMARY_KEYS.get(table_name)
    if primary_key and primary_key in cleaned.columns:
        valid_keys[table_name] = set(cleaned[primary_key].dropna())
    return cleaned


def ingest_data() -> dict[str, pd.DataFrame]:
    """Read and clean all configured tables directly from MySQL."""
    tables: dict[str, pd.DataFrame] = {}
    valid_keys: dict[str, set] = {}
    with _get_engine().connect() as connection:
        for table_name in RAW_TABLES:
            tables[table_name] = ingest_table(table_name, connection, valid_keys)
    return tables


def load_config() -> dict[str, Any]:
    """Load and normalize the label configuration."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if "reference_date" not in config:
        raise ValueError("label_config.yaml must define reference_date.")

    config["reference_date"] = pd.Timestamp(config["reference_date"])
    return config


def load_data() -> dict[str, pd.DataFrame]:
    """Load and clean all configured database tables."""
    return ingest_data()


def load_pipeline_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Return raw tables and configuration with the orders cutoff applied."""
    config = load_config()
    tables = load_data()

    orders = tables["orders"].copy()
    if ORDER_TIMESTAMP_COLUMN not in orders.columns:
        raise ValueError(
            f"orders table must contain {ORDER_TIMESTAMP_COLUMN!r}."
        )

    orders[ORDER_TIMESTAMP_COLUMN] = pd.to_datetime(
        orders[ORDER_TIMESTAMP_COLUMN],
        errors="coerce",
    )
    reference_date = config["reference_date"]
    orders = orders.loc[
        orders[ORDER_TIMESTAMP_COLUMN].notna()
        & (orders[ORDER_TIMESTAMP_COLUMN] <= reference_date)
    ].copy()

    return_window = int(config.get("return_window", 0))
    censoring_rules = config.get("censoring_rules", {})
    strategy = censoring_rules.get("strategy", "flag")
    if strategy not in {"drop", "flag"}:
        raise ValueError(f"Unsupported censoring strategy: {strategy!r}")

    days_since_purchase = (reference_date - orders[ORDER_TIMESTAMP_COLUMN]).dt.days
    censored = days_since_purchase < return_window
    if strategy == "flag":
        flag_column = censoring_rules.get("censored_flag_column", "is_censored")
        orders[flag_column] = censored
    else:
        orders = orders.loc[~censored].copy()

    tables["orders"] = orders.reset_index(drop=True)

    return tables, config


def main() -> None:
    """Load point-in-time-safe inputs directly from MySQL."""
    pipeline_tables, pipeline_config = load_pipeline_inputs()
    print(f"Loaded tables: {', '.join(pipeline_tables)}")
    print(
        "Orders before cutoff loaded: "
        f"{len(pipeline_tables['orders']):,} rows"
    )
    print(f"Reference date: {pipeline_config['reference_date'].date()}")


if __name__ == "__main__":
    main()
