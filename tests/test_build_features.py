import pandas as pd
import pytest
from unittest.mock import MagicMock

import app.Features.build_features as bf

def mock_read_query(monkeypatch, data):

    monkeypatch.setattr(
        bf,
        "read_query",
        lambda *args, **kwargs: data
    )


def test_get_reference_date():
    """
    Test that get_reference_date() returns a pandas Timestamp.
    """

    connection = MagicMock()

    connection.execute.return_value.scalar.return_value = "2018-08-01"

    bf.engine = MagicMock()

    bf.engine.connect.return_value.__enter__.return_value = connection

    result = bf.get_reference_date()

    assert isinstance(result, pd.Timestamp)
    assert result == pd.Timestamp("2018-08-01")


def test_get_reference_date_when_no_date():
    """
    Test that get_reference_date() raises ValueError
    when there is no order date in the database.
    """

    connection = MagicMock()

    connection.execute.return_value.scalar.return_value = None

    bf.engine = MagicMock()

    bf.engine.connect.return_value.__enter__.return_value = connection

    with pytest.raises(ValueError):
        bf.get_reference_date()


def test_build_customer_base(monkeypatch):
    """
    Test that build_customer_base() returns customer IDs.
    """

    data = pd.DataFrame({
        "customer_unique_id": [
            "c1",
            "c2"
        ]
    })

    mock_read_query(monkeypatch, data)

    result = bf.build_customer_base(
        pd.Timestamp("2018-08-01")
    )

    assert list(result.columns) == [
        "customer_unique_id"
    ]

    assert len(result) == 2

    assert result["customer_unique_id"].is_unique


def test_build_rfm_features(monkeypatch):
    """
    Test RFM features:

    - recency_days
    - frequency
    - monetary_value
    - avg_order_value
    """

    data = pd.DataFrame({
        "customer_unique_id": [
            "c1",
            "c2"
        ],
        "last_purchase_date": [
            "2018-07-01",
            "2018-03-01"
        ],
        "frequency": [
            2,
            4
        ],
        "monetary_value": [
            100,
            200
        ]
    })

    mock_read_query(monkeypatch, data)

    reference_date = pd.Timestamp("2018-08-01")

    result = bf.build_rfm_features(reference_date)

    assert list(result.columns) == [
        "customer_unique_id",
        "recency_days",
        "frequency",
        "monetary_value",
        "avg_order_value"
    ]

    # c1:
    # August 1 - July 1 = 31 days
    assert result.loc[
        result["customer_unique_id"] == "c1",
        "recency_days"
    ].iloc[0] == 31

    # c2:
    # March 1 - August 1 = 153 days
    assert result.loc[
        result["customer_unique_id"] == "c2",
        "recency_days"
    ].iloc[0] == 153

    # c1:
    # 100 / 2 = 50
    assert result.loc[
        result["customer_unique_id"] == "c1",
        "avg_order_value"
    ].iloc[0] == 50

    # c2:
    # 200 / 4 = 50
    assert result.loc[
        result["customer_unique_id"] == "c2",
        "avg_order_value"
    ].iloc[0] == 50

    # Customer IDs should be unique
    assert result["customer_unique_id"].is_unique

def test_build_order_features(monkeypatch):
    """
    Test that order behavior features are returned correctly.
    """

    data = pd.DataFrame({
        "customer_unique_id": [
            "c1",
            "c2"
        ],
        "delivered_orders": [
            3,
            1
        ],
        "canceled_orders": [
            0,
            1
        ],
        "shipped_orders": [
            1,
            0
        ],
        "unavailable_orders": [
            0,
            0
        ],
        "delivered_rate": [
            0.75,
            0.50
        ],
        "single_order_customer": [
            0,
            1
        ],
        "latest_order_status": [
            "delivered",
            "canceled"
        ]
    })

    mock_read_query(monkeypatch, data)

    result = bf.build_order_features(
        pd.Timestamp("2018-08-01")
    )

    assert list(result.columns) == list(data.columns)

    assert len(result) == 2

    # Customer IDs should be unique
    assert result["customer_unique_id"].is_unique

    # Check some values
    assert result.loc[
        0,
        "delivered_orders"
    ] == 3

    assert result.loc[
        1,
        "latest_order_status"
    ] == "canceled"


def test_build_payment_features(monkeypatch):
    """
    Test:

    - preferred payment type
    - average payment installments
    """

    # -----------------------------------------------------
    # First database query
    # -----------------------------------------------------

    payment_types = pd.DataFrame({
        "customer_unique_id": [
            "c1",
            "c1",
            "c2",
            "c2"
        ],
        "payment_type": [
            "credit_card",
            "boleto",
            "boleto",
            "credit_card"
        ],
        "payment_order_count": [
            3,
            1,
            2,
            2
        ]
    })

    # -----------------------------------------------------
    # Second database query
    # -----------------------------------------------------

    installments = pd.DataFrame({
        "order_id": [
            "o1",
            "o2",
            "o3",
            "o4"
        ],
        "customer_unique_id": [
            "c1",
            "c1",
            "c2",
            "c2"
        ],
        "order_avg_installments": [
            2,
            4,
            2,
            2
        ]
    })

    # build_payment_features() calls read_query()
    # two times.
    responses = [
        payment_types,
        installments
    ]

    monkeypatch.setattr(
        bf,
        "read_query",
        lambda *args, **kwargs: responses.pop(0)
    )

    result = bf.build_payment_features(
        pd.Timestamp("2018-08-01")
    )

    # -----------------------------------------------------
    # Check preferred payment type
    # -----------------------------------------------------

    # c1:
    # credit_card = 3
    # boleto = 1
    assert result.loc[
        result["customer_unique_id"] == "c1",
        "preferred_payment_type"
    ].iloc[0] == "credit_card"

    # c2:
    # boleto = 2
    # credit_card = 2
    assert result.loc[
        result["customer_unique_id"] == "c2",
        "preferred_payment_type"
    ].iloc[0] == "boleto"

    # -----------------------------------------------------
    # Check average installments
    # -----------------------------------------------------

    # c1:
    # (2 + 4) / 2 = 3
    assert result.loc[
        result["customer_unique_id"] == "c1",
        "avg_payment_installments"
    ].iloc[0] == 3

    # c2:
    # (2 + 2) / 2 = 2
    assert result.loc[
        result["customer_unique_id"] == "c2",
        "avg_payment_installments"
    ].iloc[0] == 2

    # Customer IDs should be unique
    assert result["customer_unique_id"].is_unique

def test_build_review_features(monkeypatch):
    """
    Test review features.
    """

    data = pd.DataFrame({
        "customer_unique_id": [
            "c1",
            "c2"
        ],
        "avg_review_score": [
            4.5,
            3.0
        ],
        "review_count": [
            2,
            1
        ]
    })

    mock_read_query(monkeypatch, data)

    result = bf.build_review_features(
        pd.Timestamp("2018-08-01")
    )

    assert list(result.columns) == list(data.columns)

    assert len(result) == 2

    assert result["customer_unique_id"].is_unique

    assert result.loc[
        0,
        "avg_review_score"
    ] == 4.5


def test_build_product_features(monkeypatch):
    """
    Test:
    - average items per order
    - dominant product category
    """

    # -----------------------------------------------------
    # First database query
    # -----------------------------------------------------

    product_data = pd.DataFrame({
        "customer_unique_id": [
            "c1",
            "c2"
        ],
        "total_items": [
            10,
            4
        ],
        "unique_products": [
            5,
            2
        ],
        "unique_categories": [
            2,
            1
        ],
        "order_count": [
            2,
            2
        ]
    })

    # -----------------------------------------------------
    # Second database query
    # -----------------------------------------------------

    category_data = pd.DataFrame({
        "customer_unique_id": [
            "c1",
            "c1",
            "c2",
            "c2"
        ],
        "product_category_name": [
            "electronics",
            "beauty",
            "beauty",
            "electronics"
        ],
        "category_item_count": [
            6,
            4,
            2,
            2
        ]
    })

    # The function calls read_query() twice.
    responses = [
        product_data,
        category_data
    ]

    monkeypatch.setattr(
        bf,
        "read_query",
        lambda *args, **kwargs: responses.pop(0)
    )

    result = bf.build_product_features(
        pd.Timestamp("2018-08-01")
    )

    # Check output columns
    assert list(result.columns) == [
        "customer_unique_id",
        "total_items",
        "unique_products",
        "unique_categories",
        "dominant_product_category",
        "avg_items_per_order"
    ]

    # -----------------------------------------------------
    # Check average items per order
    # -----------------------------------------------------

    # c1:
    # 10 / 2 = 5
    assert result.loc[
        result["customer_unique_id"] == "c1",
        "avg_items_per_order"
    ].iloc[0] == 5

    # c2:
    # 4 / 2 = 2
    assert result.loc[
        result["customer_unique_id"] == "c2",
        "avg_items_per_order"
    ].iloc[0] == 2

    # -----------------------------------------------------
    # Check dominant category
    # -----------------------------------------------------

    # c1:
    # electronics = 6
    # beauty = 4
    assert result.loc[
        result["customer_unique_id"] == "c1",
        "dominant_product_category"
    ].iloc[0] == "electronics"

    # c2:
    # beauty = 2
    # electronics = 2
    assert result.loc[
        result["customer_unique_id"] == "c2",
        "dominant_product_category"
    ].iloc[0] == "beauty"

    assert result["customer_unique_id"].is_unique


def test_build_fulfillment_features(monkeypatch):
    """
    Test fulfillment features.
    """

    data = pd.DataFrame({
        "customer_unique_id": [
            "c1",
            "c2"
        ],
        "avg_delivery_days": [
            5.0,
            8.0
        ]
    })

    mock_read_query(monkeypatch, data)

    result = bf.build_fulfillment_features(
        pd.Timestamp("2018-08-01")
    )

    assert list(result.columns) == list(data.columns)

    assert len(result) == 2

    assert result["customer_unique_id"].is_unique


def test_build_time_features(monkeypatch):
    """
    Test:

    - first purchase date
    - last purchase date
    - tenure days
    - active purchase days
    """

    data = pd.DataFrame({
        "customer_unique_id": [
            "c1",
            "c1",
            "c2"
        ],
        "order_purchase_timestamp": [
            "2018-01-01",
            "2018-03-15",
            "2018-04-12"
        ]
    })

    mock_read_query(monkeypatch, data)

    reference_date = pd.Timestamp("2018-06-01")

    result = bf.build_time_features(
        reference_date
    )

    # -----------------------------------------------------
    # c1 first purchase
    # -----------------------------------------------------

    assert result.loc[
        result["customer_unique_id"] == "c1",
        "first_purchase_date"
    ].iloc[0] == pd.Timestamp("2018-01-01")

    # -----------------------------------------------------
    # c1 last purchase
    # -----------------------------------------------------

    assert result.loc[
        result["customer_unique_id"] == "c1",
        "last_purchase_date"
    ].iloc[0] == pd.Timestamp("2018-03-15")

    # -----------------------------------------------------
    # c1 tenure
    #
    # June 1 - January 1 = 151 days
    # -----------------------------------------------------

    assert result.loc[
        result["customer_unique_id"] == "c1",
        "tenure_days"
    ].iloc[0] == 151



    assert result.loc[
        result["customer_unique_id"] == "c1",
        "active_purchase_days"
    ].iloc[0] == 2

    # c2 has one purchase
    assert result.loc[
        result["customer_unique_id"] == "c2",
        "active_purchase_days"
    ].iloc[0] == 1

    assert result["customer_unique_id"].is_unique


def test_build_geography_features(monkeypatch):
    """
    Test:

    - dominant city/state
    - city_state creation
    - unknown when location is missing
    """

    data = pd.DataFrame({
        "customer_unique_id": [
            "c1",
            "c1",
            "c2"
        ],
        "customer_city": [
            "sao paulo",
            "campinas",
            None
        ],
        "customer_state": [
            "SP",
            "SP",
            None
        ],

        "location_order_count": [
            5,
            2,
            1
        ]
    })

    mock_read_query(monkeypatch, data)

    result = bf.build_geography_features(
        pd.Timestamp("2018-08-01")
    )

    # c1 has:
    # sao paulo = 5 orders
    # campinas  = 2 orders

    assert result.loc[
        result["customer_unique_id"] == "c1",
        "customer_city_state"
    ].iloc[0] == "sao paulo, SP"

    # c2 has no city or state.
    # The actual code converts this to "unknown".
    assert result.loc[
        result["customer_unique_id"] == "c2",
        "customer_city_state"
    ].iloc[0] == "unknown"

    assert result["customer_unique_id"].is_unique


def test_merge_features():
    customer_base = pd.DataFrame({
        "customer_unique_id": [
            "c1",
            "c2"
        ]
    })

    reviews = pd.DataFrame({
        "customer_unique_id": [
            "c1"
        ],
        "avg_review_score": [
            4.5
        ]
    })

    result = bf.merge_features(
        customer_base,
        {
            "reviews": reviews
        }
    )

    # Both customers should remain
    assert len(result) == 2

    # c1 has review data
    assert result.loc[
        result["customer_unique_id"] == "c1",
        "avg_review_score"
    ].iloc[0] == 4.5

    # c2 has no review data
    assert pd.isna(
        result.loc[
            result["customer_unique_id"] == "c2",
            "avg_review_score"
        ].iloc[0]
    )


def test_merge_features_duplicate_customer():
    """
    Test that duplicate customer IDs inside a feature
    table raise ValueError.
    """

    customer_base = pd.DataFrame({
        "customer_unique_id": [
            "c1",
            "c2"
        ]
    })

    reviews = pd.DataFrame({
        "customer_unique_id": [
            "c1",
            "c1"
        ],
        "avg_review_score": [
            4.5,
            3.5
        ]
    })

    # Duplicate c1 should cause ValueError
    with pytest.raises(ValueError):
        bf.merge_features(
            customer_base,
            {
                "reviews": reviews
            }
        )

def test_clean_features():
    features = pd.DataFrame({
        "customer_unique_id": [
            "c1",
            "c2"
        ],

        "frequency": [
            1,
            None
        ],

        "single_order_customer": [
            1,
            None
        ],

        "preferred_payment_type": [
            "credit_card",
            None
        ],

        "first_purchase_date": [
            "2018-01-01",
            "2018-02-01"
        ]
    })

    result = bf.clean_features(features)


    assert result.loc[
        1,
        "frequency"
    ] == 0

    assert result.loc[
        1,
        "single_order_customer"
    ] == 0

    # It should be an integer column
    assert result[
        "single_order_customer"
    ].dtype == int

    assert result.loc[
        1,
        "preferred_payment_type"
    ] == "unknown"

    assert pd.api.types.is_datetime64_any_dtype(
        result["first_purchase_date"]
    )

def test_validate_features_wrong_row_count():
    """
    Test that validation fails when the number of
    feature rows is different from the customer base.
    """

    customer_base = pd.DataFrame({
        "customer_unique_id": [
            "c1",
            "c2"
        ]
    })

    features = pd.DataFrame({
        "customer_unique_id": [
            "c1"
        ]
    })

    # 2 customers in base but only 1 feature row
    with pytest.raises(ValueError):
        bf.validate_features(
            features,
            customer_base
        )


def test_validate_features_missing_columns():
    """
    Test that validation fails when required columns
    are missing.
    """

    customer_base = pd.DataFrame({
        "customer_unique_id": [
            "c1"
        ]
    })

    # Deliberately incomplete feature table
    features = pd.DataFrame({
        "customer_unique_id": [
            "c1"
        ]
    })

    # Required feature columns are missing
    with pytest.raises(ValueError):
        bf.validate_features(
            features,
            customer_base
        )
