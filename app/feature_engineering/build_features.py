from pathlib import Path

import pandas as pd
import yaml
from sqlalchemy import text

from app.database import engine


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

CONFIG_PATH = PROJECT_ROOT / "app" / "config" / "feature_config.yaml"


# ============================================================
# CONFIG
# ============================================================

def load_config():
    """Load feature engineering configuration."""

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


# ============================================================
# REFERENCE DATE
# ============================================================

def get_reference_date():
    """
    Get the latest order purchase date from the database.

    All customer features are calculated using
    complete customer history up to this date.
    """

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


# ============================================================
# CUSTOMER BASE
# ============================================================

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

    with engine.connect() as connection:
        customers = pd.read_sql(
            query,
            connection,
            params={
                "reference_date": reference_date
            }
        )

    print(
        f"Customer base created: "
        f"{len(customers):,} customers"
    )

    return customers


# ============================================================
# RFM FEATURES
# ============================================================

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

    with engine.connect() as connection:
        rfm = pd.read_sql(
            query,
            connection,
            params={
                "reference_date": reference_date
            }
        )

    rfm["last_purchase_date"] = pd.to_datetime(
        rfm["last_purchase_date"]
    )

    # Recency
    rfm["recency_days"] = (
        reference_date
        - rfm["last_purchase_date"]
    ).dt.days

    # Average value of one order
    rfm["avg_order_value"] = (
        rfm["monetary_value"]
        / rfm["frequency"]
    )

    rfm = rfm[
        [
            "customer_unique_id",
            "recency_days",
            "frequency",
            "monetary_value",
            "avg_order_value",
        ]
    ]

    print(
        f"RFM features created: "
        f"{len(rfm):,} customers"
    )

    return rfm


# ============================================================
# ORDER BEHAVIOR FEATURES
# ============================================================

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

    with engine.connect() as connection:
        orders = pd.read_sql(
            query,
            connection,
            params={
                "reference_date": reference_date
            }
        )

    print(
        f"Order behavior features created: "
        f"{len(orders):,} customers"
    )

    return orders


# ============================================================
# PAYMENT FEATURES
# ============================================================

def build_payment_features(reference_date):

    print("\nBuilding payment features...")

    # --------------------------------------------------------
    # Preferred payment type
    # --------------------------------------------------------

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

    with engine.connect() as connection:
        payment_types = pd.read_sql(
            payment_type_query,
            connection,
            params={
                "reference_date": reference_date
            }
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

    # --------------------------------------------------------
    # Average payment installments
    # --------------------------------------------------------

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

    with engine.connect() as connection:
        order_installments = pd.read_sql(
            order_payment_query,
            connection,
            params={
                "reference_date": reference_date
            }
        )

    customer_installments = (
        order_installments
        .groupby("customer_unique_id")
        ["order_avg_installments"]
        .mean()
        .reset_index()
    )

    customer_installments = (
        customer_installments
        .rename(
            columns={
                "order_avg_installments":
                "avg_payment_installments"
            }
        )
    )

    # --------------------------------------------------------
    # Merge payment features
    # --------------------------------------------------------

    payment_features = preferred_payment.merge(
        customer_installments,
        on="customer_unique_id",
        how="outer",
        validate="one_to_one"
    )

    print(
        f"Payment features created: "
        f"{len(payment_features):,} customers"
    )

    return payment_features


# ============================================================
# REVIEW FEATURES
# ============================================================

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

    with engine.connect() as connection:
        reviews = pd.read_sql(
            query,
            connection,
            params={
                "reference_date": reference_date
            }
        )

    print(
        f"Review features created: "
        f"{len(reviews):,} customers"
    )

    return reviews


# ============================================================
# PRODUCT FEATURES
# ============================================================

def build_product_features(reference_date):

    print("\nBuilding product features...")

    # --------------------------------------------------------
    # Product quantity and diversity
    # --------------------------------------------------------

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

    with engine.connect() as connection:
        products = pd.read_sql(
            product_query,
            connection,
            params={
                "reference_date": reference_date
            }
        )

    # Average number of items in one order
    products["avg_items_per_order"] = (
        products["total_items"]
        / products["order_count"]
    )

    # --------------------------------------------------------
    # Dominant product category
    # --------------------------------------------------------

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

    with engine.connect() as connection:
        categories = pd.read_sql(
            category_query,
            connection,
            params={
                "reference_date": reference_date
            }
        )

    # Highest number of purchased items first.
    # Product category name is used as a deterministic
    # tie-breaker.
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

    # --------------------------------------------------------
    # Merge dominant category
    # --------------------------------------------------------

    products = products.merge(
        dominant_category,
        on="customer_unique_id",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # Keep only required product features
    # --------------------------------------------------------

    products = products[
        [
            "customer_unique_id",
            "total_items",
            "unique_products",
            "unique_categories",
            "dominant_product_category",
            "avg_items_per_order",
        ]
    ]

    print(
        f"Product features created: "
        f"{len(products):,} customers"
    )

    return products


# ============================================================
# FULFILLMENT FEATURES
# ============================================================

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

    with engine.connect() as connection:
        fulfillment = pd.read_sql(
            query,
            connection,
            params={
                "reference_date": reference_date
            }
        )

    print(
        f"Fulfillment features created: "
        f"{len(fulfillment):,} customers"
    )

    return fulfillment


# ============================================================
# TIME BEHAVIOR FEATURES
# ============================================================

def build_time_features(reference_date):

    print("\nBuilding time behavior features...")

    # --------------------------------------------------------
    # Get every customer's purchase dates
    # --------------------------------------------------------

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

    with engine.connect() as connection:
        orders = pd.read_sql(
            query,
            connection,
            params={
                "reference_date": reference_date
            }
        )

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )

    # --------------------------------------------------------
    # First and last purchase
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Tenure
    # --------------------------------------------------------

    time_features["tenure_days"] = (
        reference_date
        - time_features["first_purchase_date"]
    ).dt.days

    # --------------------------------------------------------
    # Active purchase days
    # --------------------------------------------------------

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

    time_features = time_features.merge(
        active_days,
        on="customer_unique_id",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        f"Time features created: "
        f"{len(time_features):,} customers"
    )

    return time_features

# ============================================================
# GEOGRAPHY FEATURES
# ============================================================

def build_geography_features(reference_date):
    """
    Assign each customer their most-frequently-used city/state,
    not simply the alphabetically last one. This mirrors the
    dominant-value pattern used for preferred_payment_type and
    dominant_product_category, so a customer with mixed shipping
    addresses is labeled by where they actually order from most,
    with a deterministic (city, state) tie-breaker.
    """

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

    with engine.connect() as connection:
        geography = pd.read_sql(
            query,
            connection,
            params={
                "reference_date": reference_date
            }
        )

    # Most-used location first. City/state used as a
    # deterministic tie-breaker.
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

    # --------------------------------------------------------
    # Combine city and state
    # --------------------------------------------------------

    dominant_location["customer_city_state"] = (
        dominant_location["customer_city"].fillna("")
        + ", "
        + dominant_location["customer_state"].fillna("")
    )

    dominant_location["customer_city_state"] = (
        dominant_location["customer_city_state"]
        .str.strip()
        .str.strip(",")
    )

    dominant_location["customer_city_state"] = (
        dominant_location["customer_city_state"]
        .replace("", "unknown")
    )

    dominant_location = dominant_location[
        [
            "customer_unique_id",
            "customer_city_state",
        ]
    ]

    print(
        f"Geography features created: "
        f"{len(dominant_location):,} customers"
    )

    return dominant_location


# ============================================================
# MERGE ALL FEATURES
# ============================================================

def merge_features(customer_base, feature_tables):

    print("\nMerging all feature groups...")

    features = customer_base.copy()

    original_count = len(features)

    for name, table in feature_tables.items():

        print(f"  Merging {name}...")

        # ----------------------------------------------------
        # Check table structure
        # ----------------------------------------------------

        if "customer_unique_id" not in table.columns:

            raise ValueError(
                f"{name} does not contain "
                f"'customer_unique_id'."
            )

        # ----------------------------------------------------
        # Check duplicate customers
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Merge
        # ----------------------------------------------------

        features = features.merge(
            table,
            on="customer_unique_id",
            how="left",
            validate="one_to_one"
        )

        # ----------------------------------------------------
        # Customer count must never increase
        # ----------------------------------------------------

        if len(features) != original_count:

            raise ValueError(
                f"Customer count changed "
                f"after merging {name}."
            )

    return features


# ============================================================
# CLEAN FEATURES
# ============================================================

def clean_features(features):

    print("\nCleaning feature values...")

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    numeric_columns = [
        # RFM
        "recency_days",
        "frequency",
        "monetary_value",
        "avg_order_value",

        # Order behavior
        "delivered_orders",
        "canceled_orders",
        "shipped_orders",
        "unavailable_orders",
        "delivered_rate",
        "single_order_customer",

        # Product
        "total_items",
        "unique_products",
        "unique_categories",
        "avg_items_per_order",

        # Payment
        "avg_payment_installments",

        # Reviews
        "avg_review_score",
        "review_count",

        # Fulfillment
        "avg_delivery_days",

        # Time
        "tenure_days",
        "active_purchase_days",
    ]

    for column in numeric_columns:

        if column in features.columns:

            features[column] = pd.to_numeric(
                features[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Count features
    # --------------------------------------------------------

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

            features[column] = (
                features[column]
                .fillna(0)
            )

    # --------------------------------------------------------
    # Single order flag
    # --------------------------------------------------------

    if "single_order_customer" in features.columns:

        features["single_order_customer"] = (
            features["single_order_customer"]
            .fillna(0)
            .astype(int)
        )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------

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


# ============================================================
# VALIDATE FEATURES
# ============================================================

def validate_features(features, customer_base):

    print("\nValidating final feature table...")

    # --------------------------------------------------------
    # Customer count
    # --------------------------------------------------------

    if len(features) != len(customer_base):

        raise ValueError(
            "Final customer count does not "
            "match customer base."
        )

    # --------------------------------------------------------
    # Duplicate customers
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [

        # Customer ID
        "customer_unique_id",

        # ----------------------------------------------------
        # RFM
        # ----------------------------------------------------

        "recency_days",
        "frequency",
        "monetary_value",
        "avg_order_value",

        # ----------------------------------------------------
        # Order behavior
        # ----------------------------------------------------

        "delivered_orders",
        "canceled_orders",
        "shipped_orders",
        "unavailable_orders",
        "delivered_rate",
        "single_order_customer",
        "latest_order_status",

        # ----------------------------------------------------
        # Payment
        # ----------------------------------------------------

        "preferred_payment_type",
        "avg_payment_installments",

        # ----------------------------------------------------
        # Reviews
        # ----------------------------------------------------

        "avg_review_score",
        "review_count",

        # ----------------------------------------------------
        # Product
        # ----------------------------------------------------

        "total_items",
        "unique_products",
        "unique_categories",
        "dominant_product_category",
        "avg_items_per_order",

        # ----------------------------------------------------
        # Fulfillment
        # ----------------------------------------------------

        "avg_delivery_days",

        # ----------------------------------------------------
        # Time behavior
        # ----------------------------------------------------

        "first_purchase_date",
        "last_purchase_date",
        "tenure_days",
        "active_purchase_days",

        # ----------------------------------------------------
        # Geography
        # ----------------------------------------------------

        "customer_city_state",

        # ----------------------------------------------------
        # Reference
        # ----------------------------------------------------

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

    # --------------------------------------------------------
    # Check customer ID is not null
    # --------------------------------------------------------

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


# ============================================================
# SAVE FEATURES
# ============================================================

def save_features(features):

    output_directory = PROJECT_ROOT / "outputs"

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_directory
        / "customer_features_raw.csv"
    )

    features.to_csv(
        output_path,
        index=False
    )

    print("\nFeatures saved to:")
    print(output_path)


# ============================================================
# PRINT SUMMARY
# ============================================================

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


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("CustomerSphere")
    print("Feature Engineering")
    print("=" * 60)

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    config = load_config()

    # Avoid unused-variable warning while keeping
    # configuration loading available for future settings.
    _ = config

    # --------------------------------------------------------
    # Get reference date dynamically
    # --------------------------------------------------------

    reference_date = get_reference_date()

    print(
        f"\nReference date: "
        f"{reference_date}"
    )

    # --------------------------------------------------------
    # Build customer base
    # --------------------------------------------------------

    customer_base = build_customer_base(
        reference_date
    )

    # --------------------------------------------------------
    # Build all feature groups
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Merge all feature groups
    # --------------------------------------------------------

    features = merge_features(
        customer_base,
        feature_tables
    )

    # --------------------------------------------------------
    # Add reference date
    # --------------------------------------------------------

    features["reference_date"] = reference_date

    # --------------------------------------------------------
    # Clean features
    # --------------------------------------------------------

    features = clean_features(
        features
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    features = validate_features(
        features,
        customer_base
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_features(features)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(features)

    print(
        "\nFeature engineering completed successfully."
    )

    print(
        "Database tables were NOT modified."
    )

    print(
        "No churn label was created."
    )

    print(
        "All customers with valid orders were retained."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
