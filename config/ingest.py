import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from sqlalchemy import inspect, text

try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

try:
    from cleaning import clean_dataframe
except ImportError:
    from cleaning import clean_dataframe
    from database import engine

DATASET_DIR = os.getenv("DATASET_DIR")
if not DATASET_DIR:
    sys.exit("ERROR: set DATASET_DIR in your .env file")

# Ordered so that every table appears AFTER the tables it depends on
# (its foreign keys point backwards, never forwards).
TABLES = {
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
}

# Tables that other tables point to via foreign key, and the column that
# holds their primary key. After a table here is cleaned and loaded, the
# set of IDs that actually survived cleaning is remembered in `valid_keys`
# so that any child table referencing it can drop now-orphaned rows before
# they hit the DB's foreign key constraints.
PRIMARY_KEYS = {
    "customers": "customer_id",
    "sellers": "seller_id",
    "products": "product_id",
    "orders": "order_id",
}

def print_before_cleaning_summary(table_name, df):
    """
    Print the condition of the data before cleaning.
    """

    total_missing = df.isna().sum().sum()
    duplicate_rows = df.duplicated().sum()

    print("\n" + "=" * 60)
    print(f"{table_name.upper()} - BEFORE CLEANING")
    print("=" * 60)

    print(f"Rows:                 {len(df):,}")
    print(f"Columns:              {len(df.columns):,}")
    print(f"Missing values:       {total_missing:,}")
    print(f"Duplicate full rows:  {duplicate_rows:,}")

    print("=" * 60)


def print_after_cleaning_summary(table_name, before_df, after_df):
    """
    Compare the data before and after cleaning.
    """

    rows_before = len(before_df)
    rows_after = len(after_df)

    rows_removed = rows_before - rows_after

    missing_before = before_df.isna().sum().sum()
    missing_after = after_df.isna().sum().sum()

    print("\n" + "=" * 60)
    print(f"{table_name.upper()} - AFTER CLEANING")
    print("=" * 60)

    print(f"Rows before:          {rows_before:,}")
    print(f"Rows after:           {rows_after:,}")
    print(f"Rows removed:         {rows_removed:,}")

    print("-" * 60)

    print(f"Missing before:       {missing_before:,}")
    print(f"Missing after:        {missing_after:,}")

    print("=" * 60)


def print_validation_summary(table_name, df):
    """
    Perform basic validation on the cleaned DataFrame.
    """

    total_missing = df.isna().sum().sum()

    print("\n" + "=" * 60)
    print(f"{table_name.upper()} - VALIDATION")
    print("=" * 60)

    print(f"Rows ready for MySQL: {len(df):,}")
    print(f"Remaining NULLs:      {total_missing:,}")

    # Validate primary key if the table has one.
    if table_name in PRIMARY_KEYS:

        primary_key = PRIMARY_KEYS[table_name]

        if primary_key in df.columns:

            missing_pk = df[primary_key].isna().sum()

            duplicate_pk = df[primary_key].duplicated().sum()

            print(f"Primary key:          {primary_key}")
            print(f"Missing primary keys: {missing_pk:,}")
            print(f"Duplicate primary keys: {duplicate_pk:,}")

            if missing_pk == 0 and duplicate_pk == 0:
                print("Primary key validation: PASSED")
            else:
                print("Primary key validation: FAILED")

    print("=" * 60)
def clear_tables(table_names):
    """Delete rows from tables in FK-safe order."""

    # users references customers, but users is not part of the CSV
    # ingestion list, so it must be cleared explicitly first.
    extra_child_tables = ["users"]
    all_to_clear = extra_child_tables + list(reversed(table_names))

    with engine.begin() as conn:
        inspector = inspect(conn)
        for table_name in all_to_clear:
            if inspector.has_table(table_name):
                print(f"Clearing {table_name}...")
                conn.execute(text(f"DELETE FROM {table_name}"))
            else:
                print(f"Skipping {table_name} (table does not exist yet)...")



def _resolve_csv_path(dataset_dir, filename):
    primary = os.path.join(dataset_dir, filename)
    if os.path.exists(primary):
        return primary
    candidates = [
        filename.replace(".csv", "_clean.csv"),
        filename.replace("_clean.csv", ".csv"),
    ]
    for cand in candidates:
        alt = os.path.join(dataset_dir, cand)
        if os.path.exists(alt):
            return alt
    return primary


def load_table(table_name, valid_keys):
        """
        Read a CSV file, clean it, validate it, and load it into MySQL.

        Normal tables:
            - Show detailed reports once.

        Geolocation:
            - Process in chunks to save memory.
            - Collect statistics across all chunks.
            - Show only ONE overall report at the end.
        """

        csv_path = _resolve_csv_path(
            DATASET_DIR,
            TABLES[table_name]
        )

        print(f"\nLoading {table_name} from {csv_path}...")

        # Geolocation is large, so process it in chunks.
        chunksize = 10000 if table_name == "geolocation" else None

        reader = pd.read_csv(
            csv_path,
            chunksize=chunksize
        )

        # For normal tables, wrap the DataFrame in a list so that
        # both normal tables and chunked tables use the same loop.
        chunks = reader if chunksize else [reader]

        # =========================================================
        # OVERALL STATISTICS
        # =========================================================

        total_read = 0
        total_loaded = 0

        total_columns = None
        total_missing_before = 0
        total_missing_after = 0
        total_duplicate_rows = 0

        # Store surviving primary keys for parent tables.
        surviving_keys = (
            set()
            if table_name in PRIMARY_KEYS
            else None
        )

        # =========================================================
        # PROCESS CHUNKS
        # =========================================================

        for chunk in chunks:

            # -------------------------------
            # BEFORE CLEANING STATISTICS
            # -------------------------------

            total_read += len(chunk)

            if total_columns is None:
                total_columns = len(chunk.columns)

            total_missing_before += (
                chunk.isna().sum().sum()
            )

            total_duplicate_rows += (
                chunk.duplicated().sum()
            )

            # For normal tables, show the detailed report now.
            if not chunksize:
                print_before_cleaning_summary(
                    table_name,
                    chunk
                )

            # -------------------------------
            # CLEANING
            # -------------------------------

            cleaned = clean_dataframe(
                table_name,
                chunk,
                valid_keys=valid_keys
            )

            total_loaded += len(cleaned)

            # -------------------------------
            # AFTER CLEANING STATISTICS
            # -------------------------------

            total_missing_after += (
                cleaned.isna().sum().sum()
            )

            # For normal tables, show detailed reports now.
            if not chunksize:
                print_after_cleaning_summary(
                    table_name,
                    chunk,
                    cleaned
                )

                print_validation_summary(
                    table_name,
                    cleaned
                )

            # -------------------------------
            # SAVE SURVIVING PRIMARY KEYS
            # -------------------------------

            if surviving_keys is not None:

                primary_key = PRIMARY_KEYS[table_name]

                if primary_key in cleaned.columns:
                    surviving_keys.update(
                        cleaned[primary_key].dropna()
                    )

            # -------------------------------
            # LOAD INTO MYSQL
            # -------------------------------

            if not cleaned.empty:
                cleaned.to_sql(
                    table_name,
                    con=engine,
                    if_exists="append",
                    index=False,
                )

        # =========================================================
        # SAVE VALID PRIMARY KEYS
        # =========================================================

        if surviving_keys is not None:
            valid_keys[table_name] = surviving_keys

        # =========================================================
        # GEOLOCATION OVERALL REPORT
        # =========================================================

        if chunksize:
            print("\n" + "=" * 60)
            print(
                f"{table_name.upper()} - BEFORE CLEANING (OVERALL)"
            )
            print("=" * 60)

            print(f"Rows:                 {total_read:,}")
            print(f"Columns:              {total_columns:,}")
            print(
                f"Missing values:       "
                f"{total_missing_before:,}"
            )
            print(
                f"Duplicate full rows:  "
                f"{total_duplicate_rows:,}"
            )

            print("=" * 60)

            # -------------------------------
            # AFTER CLEANING
            # -------------------------------

            rows_removed = (
                    total_read - total_loaded
            )

            print("\n" + "=" * 60)
            print(
                f"{table_name.upper()} - AFTER CLEANING (OVERALL)"
            )
            print("=" * 60)

            print(
                f"Rows before:          "
                f"{total_read:,}"
            )

            print(
                f"Rows after:           "
                f"{total_loaded:,}"
            )

            print(
                f"Rows removed:         "
                f"{rows_removed:,}"
            )

            print("-" * 60)

            print(
                f"Missing before:       "
                f"{total_missing_before:,}"
            )

            print(
                f"Missing after:        "
                f"{total_missing_after:,}"
            )

            print("=" * 60)

            # -------------------------------
            # VALIDATION
            # -------------------------------

            print("\n" + "=" * 60)
            print(
                f"{table_name.upper()} - VALIDATION (OVERALL)"
            )
            print("=" * 60)

            print(
                f"Rows ready for MySQL: "
                f"{total_loaded:,}"
            )

            print(
                f"Remaining NULLs:      "
                f"{total_missing_after:,}"
            )

            print("Validation: COMPLETED")

            print("=" * 60)

        # =========================================================
        # FINAL SUMMARY
        # =========================================================

        dropped = total_read - total_loaded

        print("\n" + "=" * 60)
        print(
            f"FINAL SUMMARY: {table_name.upper()}"
        )
        print("=" * 60)

        print(f"Total rows read:       {total_read:,}")
        print(f"Total rows removed:    {dropped:,}")
        print(f"Total rows loaded:     {total_loaded:,}")

        if total_read > 0:
            removal_percentage = (
                                         dropped / total_read
                                 ) * 100

            print(
                f"Removal percentage:    "
                f"{removal_percentage:.2f}%"
            )

        print("=" * 60)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"Usage: python -m app.ingest <table_name|all> [--replace]\n"
                  f"Tables: {', '.join(TABLES)}")

    target = sys.argv[1]
    replace = "--replace" in sys.argv

    # Accumulates {table_name: set_of_surviving_ids} as parent tables load,
    # so child tables can filter out rows orphaned by upstream cleaning.
    valid_keys = {}

    if target == "all":
        names = list(TABLES)
        if replace:
            clear_tables(names)  # clears in reverse (children first)
        for name in names:
            load_table(name, valid_keys)
    else:
        if target not in TABLES:
            sys.exit(f"Unknown table: {target}")
        if replace:
            clear_tables([target])
        # Loading a single non-"all" table has no upstream valid_keys context,
        # so FK-orphan filtering is skipped — only safe if its parent tables
        # are already fully loaded and unchanged in the DB.
        load_table(target, valid_keys)