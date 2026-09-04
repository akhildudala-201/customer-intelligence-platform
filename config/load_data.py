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
ORDER_TIMESTAMP_COLUMN = "order_purchase_timestamp"


def _get_engine():
    """Create an engine for the configured MySQL database."""
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


def load_config() -> dict[str, Any]:
    """Load and normalize the label configuration."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if "reference_date" not in config:
        raise ValueError("label configuration must define reference_date")

    config["reference_date"] = pd.Timestamp(config["reference_date"])
    return config


def load_data() -> dict[str, pd.DataFrame]:
    """Load all configured tables directly from MySQL."""
    tables: dict[str, pd.DataFrame] = {}
    with _get_engine().connect() as connection:
        for table_name in RAW_TABLES:
            tables[table_name] = pd.read_sql_query(
                text(f"SELECT * FROM {table_name}"),
                connection,
            )
    return tables


def load_pipeline_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load database tables and apply the configured order cutoff checks."""
    config = load_config()
    tables = load_data()
    orders = tables["orders"].copy()

    if ORDER_TIMESTAMP_COLUMN not in orders.columns:
        raise ValueError(
            f"orders table must contain {ORDER_TIMESTAMP_COLUMN!r}"
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
    print(f"Orders after cutoff: {len(pipeline_tables['orders']):,} rows")
    print(f"Reference date: {pipeline_config['reference_date'].date()}")


if __name__ == "__main__":
    main()
