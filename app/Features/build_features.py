import importlib.util
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
APP_ROOT = BASE_DIR.parent

for root in (str(PROJECT_ROOT), str(APP_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_local_engine():
    """Load the project database engine without depending on a conflicting installed app package."""
    candidates = [
        "app.Database.database",
        "Database.database",
    ]

    for module_name in candidates:
        try:
            module = __import__(module_name, fromlist=["engine"])
            return module.engine
        except (ImportError, ModuleNotFoundError):
            continue

    database_path = APP_ROOT / "Database" / "database.py"
    spec = importlib.util.spec_from_file_location("local_database_module", database_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load database module from {database_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.engine


engine = _load_local_engine()

def read_query(query, reference_date):
    """Execute a parameterized SQL query and return the result as a DataFrame."""
    with engine.connect() as connection:
        return pd.read_sql(
            query,
            connection,
            params={"reference_date": reference_date}
        )

def get_reference_date():

    query = text("""
        SELECT MAX(order_purchase_timestamp)
        FROM orders
        WHERE order_purchase_timestamp IS NOT NULL
    """)

    with engine.connect() as connection:
        reference_date = connection.execute(query).scalar()

    if reference_date is None:
        raise ValueError(
            "Could not find a valid reference date in orders table."
        )

    return pd.to_datetime(reference_date)

def build_customer_base(reference_date):

    print("\nBuilding customer base...")

    query = text("""
        SELECT DISTINCT
            c.customer_unique_id

        FROM customers c

        INNER JOIN orders o
            ON c.customer_id = o.customer_id

        WHERE
            c.customer_unique_id IS NOT NULL
            AND o.order_purchase_timestamp IS NOT NULL
            AND o.order_purchase_timestamp <= :reference_date
    """)

    customers = read_query(
        query,
        reference_date
    )

    print(
        f"Customer base created: "
        f"{len(customers):,} customers"
    )

    return customers

def build_rfm_features(reference_date):

    print("\nBuilding RFM features...")

    query = text("""
        SELECT
            c.customer_unique_id,

            MAX(
                o.order_purchase_timestamp
            ) AS last_purchase_date,

            COUNT(
                DISTINCT o.order_id
            ) AS frequency,

            SUM(
                COALESCE(oi.price, 0)
                +
                COALESCE(oi.freight_value, 0)
            ) AS monetary_value

        FROM customers c

        INNER JOIN orders o
            ON c.customer_id = o.customer_id

        LEFT JOIN order_items oi
            ON o.order_id = oi.order_id

        WHERE
            c.customer_unique_id IS NOT NULL
            AND o.order_purchase_timestamp IS NOT NULL
            AND o.order_purchase_timestamp <= :reference_date

        GROUP BY
            c.customer_unique_id
    """)

    rfm = read_query(
        query,
        reference_date
    )

    rfm["last_purchase_date"] = pd.to_datetime(
        rfm["last_purchase_date"]
    )

    rfm["recency_days"] = (
        reference_date
        - rfm["last_purchase_date"]
    ).dt.days

    rfm["avg_order_value"] = (
        rfm["monetary_value"]
        / rfm["frequency"]
    )

    return rfm[
        [
            "customer_unique_id",
            "recency_days",
            "frequency",
            "monetary_value",
            "avg_order_value",
        ]
    ]

def build_order_features(reference_date):

    print("\nBuilding order behavior features...")

    query = text("""
        SELECT
            c.customer_unique_id,

            COUNT(
                DISTINCT CASE
                    WHEN o.order_status = 'delivered'
                    THEN o.order_id
                END
            ) AS delivered_orders,

            COUNT(
                DISTINCT CASE
                    WHEN o.order_status = 'canceled'
                    THEN o.order_id
                END
            ) AS canceled_orders,

            COUNT(
                DISTINCT CASE
                    WHEN o.order_status = 'shipped'
                    THEN o.order_id
                END
            ) AS shipped_orders,

            COUNT(
                DISTINCT CASE
                    WHEN o.order_status = 'unavailable'
                    THEN o.order_id
                END
            ) AS unavailable_orders,

            COUNT(
                DISTINCT CASE
                    WHEN o.order_status = 'delivered'
                    THEN o.order_id
                END
            )
            /
            COUNT(DISTINCT o.order_id)
            AS delivered_rate,

            CASE
                WHEN COUNT(DISTINCT o.order_id) = 1
                THEN 1
                ELSE 0
            END AS single_order_customer,

            SUBSTRING_INDEX(
                GROUP_CONCAT(
                    o.order_status
                    ORDER BY
                        o.order_purchase_timestamp DESC,
                        o.order_id DESC
                    SEPARATOR ','
                ),
                ',',
                1
            ) AS latest_order_status

        FROM customers c

        INNER JOIN orders o
            ON c.customer_id = o.customer_id

        WHERE
            c.customer_unique_id IS NOT NULL
            AND o.order_purchase_timestamp IS NOT NULL
            AND o.order_purchase_timestamp <= :reference_date

        GROUP BY
            c.customer_unique_id
    """)

    return read_query(
        query,
        reference_date
    )

def build_payment_features(reference_date):

    print("\nBuilding payment features...")

    payment_type_query = text("""
        SELECT
            c.customer_unique_id,
            op.payment_type,

            COUNT(
                DISTINCT o.order_id
            ) AS payment_order_count

        FROM customers c

        INNER JOIN orders o
            ON c.customer_id = o.customer_id

        INNER JOIN order_payments op
            ON o.order_id = op.order_id

        WHERE
            c.customer_unique_id IS NOT NULL
            AND o.order_purchase_timestamp IS NOT NULL
            AND o.order_purchase_timestamp <= :reference_date
            AND op.payment_type IS NOT NULL

        GROUP BY
            c.customer_unique_id,
            op.payment_type
    """)

    payment_types = read_query(
        payment_type_query,
        reference_date
    )

    payment_types = payment_types.sort_values(
        by=[
            "customer_unique_id",
            "payment_order_count",
            "payment_type",
        ],
        ascending=[
            True,
            False,
            True,
        ]
    )

    preferred_payment = (
        payment_types
        .drop_duplicates(
            subset=["customer_unique_id"]
        )
        [
            [
                "customer_unique_id",
                "payment_type",
            ]
        ]
        .rename(
            columns={
                "payment_type":
                "preferred_payment_type"
            }
        )
    )

    order_payment_query = text("""
        SELECT
            o.order_id,
            c.customer_unique_id,

            AVG(
                op.payment_installments
            ) AS order_avg_installments

        FROM customers c

        INNER JOIN orders o
            ON c.customer_id = o.customer_id

        INNER JOIN order_payments op
            ON o.order_id = op.order_id

        WHERE
            c.customer_unique_id IS NOT NULL
            AND o.order_purchase_timestamp IS NOT NULL
            AND o.order_purchase_timestamp <= :reference_date

        GROUP BY
            o.order_id,
            c.customer_unique_id
    """)

    order_installments = read_query(
        order_payment_query,
        reference_date
    )

    customer_installments = (
        order_installments
        .groupby("customer_unique_id")
        ["order_avg_installments"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "order_avg_installments":
                "avg_payment_installments"
            }
        )
    )

    return preferred_payment.merge(
        customer_installments,
        on="customer_unique_id",
        how="outer",
        validate="one_to_one"
    )

def build_review_features(reference_date):

    print("\nBuilding review features...")

    query = text("""
        SELECT
            c.customer_unique_id,

            AVG(
                r.review_score
            ) AS avg_review_score,

            COUNT(
                r.review_id
            ) AS review_count

        FROM customers c

        INNER JOIN orders o
            ON c.customer_id = o.customer_id

        INNER JOIN order_reviews r
            ON o.order_id = r.order_id

        WHERE
            c.customer_unique_id IS NOT NULL
            AND o.order_purchase_timestamp IS NOT NULL
            AND o.order_purchase_timestamp <= :reference_date

            AND (
                r.review_creation_date IS NULL
                OR r.review_creation_date <= :reference_date
            )

        GROUP BY
            c.customer_unique_id
    """)

    return read_query(
        query,
        reference_date
    )

def build_product_features(reference_date):

    print("\nBuilding product features...")

    product_query = text("""
        SELECT
            c.customer_unique_id,

            COUNT(
                oi.order_item_id
            ) AS total_items,

            COUNT(
                DISTINCT oi.product_id
            ) AS unique_products,

            COUNT(
                DISTINCT p.product_category_name
            ) AS unique_categories,

            COUNT(
                DISTINCT o.order_id
            ) AS order_count

        FROM customers c

        INNER JOIN orders o
            ON c.customer_id = o.customer_id

        INNER JOIN order_items oi
            ON o.order_id = oi.order_id

        LEFT JOIN products p
            ON oi.product_id = p.product_id

        WHERE
            c.customer_unique_id IS NOT NULL
            AND o.order_purchase_timestamp IS NOT NULL
            AND o.order_purchase_timestamp <= :reference_date

        GROUP BY
            c.customer_unique_id
    """)

    products = read_query(
        product_query,
        reference_date
    )

    products["avg_items_per_order"] = (
        products["total_items"]
        / products["order_count"]
    )

    category_query = text("""
        SELECT
            c.customer_unique_id,

            p.product_category_name,

            COUNT(
                oi.order_item_id
            ) AS category_item_count

        FROM customers c

        INNER JOIN orders o
            ON c.customer_id = o.customer_id

        INNER JOIN order_items oi
            ON o.order_id = oi.order_id

        INNER JOIN products p
            ON oi.product_id = p.product_id

        WHERE
            c.customer_unique_id IS NOT NULL
            AND o.order_purchase_timestamp IS NOT NULL
            AND o.order_purchase_timestamp <= :reference_date
            AND p.product_category_name IS NOT NULL

        GROUP BY
            c.customer_unique_id,
            p.product_category_name
    """)

    categories = read_query(
        category_query,
        reference_date
    )

    categories = categories.sort_values(
        by=[
            "customer_unique_id",
            "category_item_count",
            "product_category_name",
        ],
        ascending=[
            True,
            False,
            True,
        ]
    )

    dominant_category = (
        categories
        .drop_duplicates(
            subset=["customer_unique_id"]
        )
        [
            [
                "customer_unique_id",
                "product_category_name",
            ]
        ]
        .rename(
            columns={
                "product_category_name":
                "dominant_product_category"
            }
        )
    )

    products = products.merge(
        dominant_category,
        on="customer_unique_id",
        how="left",
        validate="one_to_one"
    )

    return products[
        [
            "customer_unique_id",
            "total_items",
            "unique_products",
            "unique_categories",
            "dominant_product_category",
            "avg_items_per_order",
        ]
    ]

def build_fulfillment_features(reference_date):

    print("\nBuilding fulfillment features...")

    query = text("""
        SELECT
            c.customer_unique_id,

            AVG(
                DATEDIFF(
                    o.order_delivered_customer_date,
                    o.order_purchase_timestamp
                )
            ) AS avg_delivery_days

        FROM customers c

        INNER JOIN orders o
            ON c.customer_id = o.customer_id

        WHERE
            c.customer_unique_id IS NOT NULL
            AND o.order_purchase_timestamp IS NOT NULL
            AND o.order_purchase_timestamp <= :reference_date
            AND o.order_status = 'delivered'
            AND o.order_delivered_customer_date IS NOT NULL
            AND o.order_delivered_customer_date <= :reference_date

        GROUP BY
            c.customer_unique_id
    """)

    return read_query(
        query,
        reference_date
    )

def build_time_features(reference_date):

    print("\nBuilding time behavior features...")

    query = text("""
        SELECT
            c.customer_unique_id,
            o.order_purchase_timestamp

        FROM customers c

        INNER JOIN orders o
            ON c.customer_id = o.customer_id

        WHERE
            c.customer_unique_id IS NOT NULL
            AND o.order_purchase_timestamp IS NOT NULL
            AND o.order_purchase_timestamp <= :reference_date

        ORDER BY
            c.customer_unique_id,
            o.order_purchase_timestamp
    """)

    orders = read_query(
        query,
        reference_date
    )

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )

    time_features = (
        orders
        .groupby("customer_unique_id")
        ["order_purchase_timestamp"]
        .agg(
            first_purchase_date="min",
            last_purchase_date="max"
        )
        .reset_index()
    )

    time_features["tenure_days"] = (
        reference_date
        - time_features["first_purchase_date"]
    ).dt.days

    active_days = (
        orders
        .assign(
            purchase_date=
            orders["order_purchase_timestamp"].dt.date
        )
        .groupby("customer_unique_id")
        ["purchase_date"]
        .nunique()
        .reset_index()
        .rename(
            columns={
                "purchase_date":
                "active_purchase_days"
            }
        )
    )

    return time_features.merge(
        active_days,
        on="customer_unique_id",
        how="left",
        validate="one_to_one"
    )

def build_geography_features(reference_date):

    print("\nBuilding geography features...")

    query = text("""
        SELECT
            c.customer_unique_id,
            c.customer_city,
            c.customer_state,

            COUNT(
                DISTINCT o.order_id
            ) AS location_order_count

        FROM customers c

        INNER JOIN orders o
            ON c.customer_id = o.customer_id

        WHERE
            c.customer_unique_id IS NOT NULL
            AND o.order_purchase_timestamp IS NOT NULL
            AND o.order_purchase_timestamp <= :reference_date

        GROUP BY
            c.customer_unique_id,
            c.customer_city,
            c.customer_state
    """)

    geography = read_query(
        query,
        reference_date
    )

    geography = geography.sort_values(
        by=[
            "customer_unique_id",
            "location_order_count",
            "customer_city",
            "customer_state",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ]
    )

    dominant_location = (
        geography
        .drop_duplicates(
            subset=["customer_unique_id"]
        )
        [
            [
                "customer_unique_id",
                "customer_city",
                "customer_state",
            ]
        ]
    )

    dominant_location["customer_city_state"] = (
        dominant_location["customer_city"].fillna("")
        + ", "
        + dominant_location["customer_state"].fillna("")
    )

    dominant_location["customer_city_state"] = (
        dominant_location["customer_city_state"]
        .str.strip()
        .str.strip(",")
        .replace("", "unknown")
    )

    return dominant_location[
        [
            "customer_unique_id",
            "customer_city_state",
        ]
    ]

def merge_features(customer_base, feature_tables):

    print("\nMerging all feature groups...")

    features = customer_base.copy()
    original_count = len(features)

    for name, table in feature_tables.items():

        print(f"  Merging {name}...")

        if "customer_unique_id" not in table.columns:
            raise ValueError(
                f"{name} does not contain 'customer_unique_id'."
            )

        duplicate_count = (
            table["customer_unique_id"]
            .duplicated()
            .sum()
        )

        if duplicate_count > 0:
            raise ValueError(
                f"{name} contains "
                f"{duplicate_count} duplicate customers."
            )

        features = features.merge(
            table,
            on="customer_unique_id",
            how="left",
            validate="one_to_one"
        )

        if len(features) != original_count:
            raise ValueError(
                f"Customer count changed after merging {name}."
            )

    return features

def clean_features(features):

    print("\nCleaning feature values...")

    numeric_columns = [
        "recency_days",
        "frequency",
        "monetary_value",
        "avg_order_value",
        "delivered_orders",
        "canceled_orders",
        "shipped_orders",
        "unavailable_orders",
        "delivered_rate",
        "single_order_customer",
        "avg_payment_installments",
        "avg_review_score",
        "review_count",
        "total_items",
        "unique_products",
        "unique_categories",
        "avg_items_per_order",
        "avg_delivery_days",
        "tenure_days",
        "active_purchase_days",
    ]

    for column in numeric_columns:
        if column in features.columns:
            features[column] = pd.to_numeric(
                features[column],
                errors="coerce"
            )

    count_columns = [
        "frequency",
        "delivered_orders",
        "canceled_orders",
        "shipped_orders",
        "unavailable_orders",
        "total_items",
        "review_count",
        "unique_products",
        "unique_categories",
        "active_purchase_days",
    ]

    for column in count_columns:
        if column in features.columns:
            features[column] = features[column].fillna(0)

    if "single_order_customer" in features.columns:
        features["single_order_customer"] = (
            features["single_order_customer"]
            .fillna(0)
            .astype(int)
        )

    categorical_columns = [
        "latest_order_status",
        "preferred_payment_type",
        "dominant_product_category",
        "customer_city_state",
    ]

    for column in categorical_columns:
        if column in features.columns:
            features[column] = (
                features[column]
                .fillna("unknown")
            )

    date_columns = [
        "first_purchase_date",
        "last_purchase_date",
    ]

    for column in date_columns:
        if column in features.columns:
            features[column] = pd.to_datetime(
                features[column],
                errors="coerce"
            )

    return features

def validate_features(features, customer_base):

    print("\nValidating final feature table...")

    if len(features) != len(customer_base):
        raise ValueError(
            "Final customer count does not match customer base."
        )

    duplicate_count = (
        features["customer_unique_id"]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Final feature table contains "
            f"{duplicate_count} duplicate customers."
        )

    required_columns = [
        "customer_unique_id",
        "recency_days",
        "frequency",
        "monetary_value",
        "avg_order_value",
        "delivered_orders",
        "canceled_orders",
        "shipped_orders",
        "unavailable_orders",
        "delivered_rate",
        "single_order_customer",
        "latest_order_status",
        "preferred_payment_type",
        "avg_payment_installments",
        "avg_review_score",
        "review_count",
        "total_items",
        "unique_products",
        "unique_categories",
        "dominant_product_category",
        "avg_items_per_order",
        "avg_delivery_days",
        "first_purchase_date",
        "last_purchase_date",
        "tenure_days",
        "active_purchase_days",
        "customer_city_state",
        "reference_date",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in features.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_columns)
        )

    missing_customer_ids = (
        features["customer_unique_id"]
        .isna()
        .sum()
    )

    if missing_customer_ids > 0:
        raise ValueError(
            f"{missing_customer_ids} customers "
            f"have missing customer_unique_id."
        )

    print("Validation successful.")

    return features

def save_features_to_db(features, table_name="customer_features"):
    """Persist the engineered feature matrix to MySQL."""

    db_features = features.copy()

    for column in db_features.columns:
        if pd.api.types.is_datetime64_any_dtype(db_features[column]):
            db_features[column] = pd.to_datetime(
                db_features[column],
                errors="coerce",
            )

    with engine.begin() as connection:
        db_features.to_sql(
            table_name,
            con=connection,
            if_exists="replace",
            index=False,
            chunksize=5000,
        )

    print(f"\nSaved feature table to MySQL table `{table_name}` ({len(db_features):,} rows).")


def print_summary(features):

    print("\n")
    print("=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)

    print(
        f"\nCustomers: {len(features):,}"
    )

    print(
        f"Total columns: {len(features.columns)}"
    )

    print("\nFinal feature columns:")

    for column in features.columns:
        print(f"  - {column}")

    print("\n")
    print("=" * 60)


def main():
    print("\n")
    print("=" * 60)
    print("CustomerSphere")
    print("Feature Engineering")
    print("=" * 60)

    reference_date = get_reference_date()

    print(
        f"\nReference date: "
        f"{reference_date}"
    )

    customer_base = build_customer_base(
        reference_date
    )

    feature_tables = {}

    feature_tables["rfm"] = (
        build_rfm_features(
            reference_date
        )
    )

    feature_tables["order_behavior"] = (
        build_order_features(
            reference_date
        )
    )

    feature_tables["payment"] = (
        build_payment_features(
            reference_date
        )
    )

    feature_tables["reviews"] = (
        build_review_features(
            reference_date
        )
    )

    feature_tables["products"] = (
        build_product_features(
            reference_date
        )
    )

    feature_tables["fulfillment"] = (
        build_fulfillment_features(
            reference_date
        )
    )

    feature_tables["time_behavior"] = (
        build_time_features(
            reference_date
        )
    )

    feature_tables["geography"] = (
        build_geography_features(
            reference_date
        )
    )

    features = merge_features(
        customer_base,
        feature_tables
    )

    features["reference_date"] = reference_date

    features = clean_features(
        features
    )

    features = validate_features(
        features,
        customer_base
    )

    save_features_to_db(features)

    print_summary(features)

    print(
        "\nFeature engineering completed successfully."
    )

    print(
        "The engineered feature matrix was saved to the MySQL database."
    )

    print(
        "Table name: customer_features"
    )

    print(
        "All customers with valid orders were retained."
    )

if __name__ == "__main__":
    main()
