import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


@pytest.fixture
def sample_customers_df():
    return pd.DataFrame({
        "customer_id": ["c1", "c2", "c3", "c4", "  c5  ", "c1"],
        "customer_unique_id": ["u1", "u2", "u3", "u4", "u5", "u1"],
        "customer_zip_code_prefix": [1001, 1002, 1003, 1004, 1005, 1001],
        "customer_city": ["sao paulo", "rio", "  curitiba  ", "", "porto alegre", "sao paulo"],
        "customer_state": ["SP", "RJ", "PR", "SP", "RS", "SP"],
    })


@pytest.fixture
def sample_orders_df():
    return pd.DataFrame({
        "order_id": ["o1", "o2", "o3", "o4", "o5", "o1"],
        "customer_id": ["c1", "c2", "c3", "c999", "c5", "c1"],
        "order_status": ["delivered", "shipped", "canceled", "delivered", "delivered", "delivered"],
        "order_purchase_timestamp": [
            "2018-01-01 10:00:00",
            "2018-02-01 10:00:00",
            "2018-03-01 10:00:00",
            "2018-04-01 10:00:00",
            "2018-05-01 10:00:00",
            "2018-01-01 10:00:00",
        ],
        "order_delivered_customer_date": [
            "2018-01-08 10:00:00",
            None,
            None,
            "2018-04-10 10:00:00",
            "2018-05-06 10:00:00",
            "2018-01-08 10:00:00",
        ],
    })


@pytest.fixture
def sample_order_items_df():
    return pd.DataFrame({
        "order_id": ["o1", "o2", "o3", "o5"],
        "order_item_id": [1, 1, 1, 1],
        "product_id": ["p1", "p2", "p3", "p5"],
        "seller_id": ["s1", "s2", "s3", "s5"],
        "price": [100.0, 50.0, -10.0, 200.0],
        "freight_value": [15.0, 10.0, 5.0, 25.0],
    })


@pytest.fixture
def sample_features_with_labels_df():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2017-01-01", periods=n, freq="3D")
    return pd.DataFrame({
        "customer_unique_id": [f"user_{i}" for i in range(n)],
        "recency_days": np.random.uniform(5, 400, size=n),
        "frequency": np.random.randint(1, 10, size=n),
        "monetary_value": np.random.uniform(20, 1000, size=n),
        "avg_order_value": np.random.uniform(20, 300, size=n),
        "delivered_orders": np.random.randint(1, 10, size=n),
        "canceled_orders": np.zeros(n, dtype=int),
        "shipped_orders": np.zeros(n, dtype=int),
        "unavailable_orders": np.zeros(n, dtype=int),
        "delivered_rate": np.ones(n),
        "single_order_customer": np.random.choice([0, 1], size=n),
        "latest_order_status": np.random.choice(["delivered", "shipped", "canceled"], size=n),
        "preferred_payment_type": np.random.choice(["credit_card", "boleto", "voucher"], size=n),
        "avg_payment_installments": np.random.uniform(1, 10, size=n),
        "avg_review_score": np.random.uniform(1, 5, size=n),
        "review_count": np.random.randint(1, 5, size=n),
        "total_items": np.random.randint(1, 10, size=n),
        "unique_products": np.random.randint(1, 5, size=n),
        "unique_categories": np.random.randint(1, 3, size=n),
        "dominant_product_category": np.random.choice(["health_beauty", "bed_bath_table", "computers"], size=n),
        "avg_items_per_order": np.random.uniform(1, 3, size=n),
        "avg_delivery_days": np.random.uniform(2, 30, size=n),
        "first_purchase_date": dates,
        "last_purchase_date": dates + pd.Timedelta(days=10),
        "tenure_days": np.random.uniform(10, 500, size=n),
        "active_purchase_days": np.random.randint(1, 5, size=n),
        "customer_city_state": np.random.choice(["sao paulo, SP", "rio de janeiro, RJ", "curitiba, PR"], size=n),
        "reference_date": pd.Timestamp("2018-10-17"),
        "churn_label": np.random.choice([0, 1, None], size=n, p=[0.3, 0.5, 0.2]),
        "censored": np.random.choice([False, True], size=n, p=[0.8, 0.2]),
    })
