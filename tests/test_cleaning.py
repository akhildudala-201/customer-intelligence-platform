import pandas as pd
import pytest
from app.Database.cleaning import (
    _strip_strings,
    _coerce_numeric,
    _dedupe,
    _drop_missing_required,
    _drop_negative,
    _clamp_out_of_range,
    _drop_high_null_rows,
    _drop_orphan_fk,
    clean_dataframe,
)

def test_strip_strings():
    df = pd.DataFrame({
        "name": [" Arun ", "  Bhuvan", "", None],
    })
    result = _strip_strings(df)
    assert result.loc[0, "name"] == "Arun"
    assert result.loc[1, "name"] == "Bhuvan"
    assert pd.isna(result.loc[2, "name"])
    assert pd.isna(result.loc[3, "name"])

def test_coerce_numeric():
    df = pd.DataFrame({
        "price": ["100", "25.5", "invalid", None],
    })
    result = _coerce_numeric(df, ["price"])
    assert result["price"].iloc[0] == 100
    assert result["price"].iloc[1] == 25.5
    assert pd.isna(result["price"].iloc[2])
    assert pd.isna(result["price"].iloc[3])

def test_dedupe():
    df = pd.DataFrame({
        "customer_id": [1, 1, 2],
        "name": ["A", "A", "B"],
    })
    result = _dedupe(df, "customers", subset=["customer_id"])
    assert len(result) == 2
    assert result["customer_id"].tolist() == [1, 2]

def test_drop_missing_required():
    df = pd.DataFrame({
        "customer_id": [1, None, 3],
        "name": ["A", "B", "C"],
    })
    result = _drop_missing_required(
        df,
        "customers",
        ["customer_id"],
    )
    assert len(result) == 2
    assert result["customer_id"].tolist() == [1, 3]

def test_drop_negative():
    df = pd.DataFrame({
        "price": [10, -5, 0, None],
    })
    result = _drop_negative(df, "products", ["price"])
    assert len(result) == 3
    assert result["price"].iloc[0] == 10
    assert result["price"].iloc[1] == 0
    assert pd.isna(result["price"].iloc[2])

def test_clamp_out_of_range():
    df = pd.DataFrame({
        "review_score": [1, 3, 5, 0, 6, None],
    })
    result = _clamp_out_of_range(
        df,
        "order_reviews",
        "review_score",
        1,
        5,
    )
    assert result.loc[0, "review_score"] == 1
    assert result.loc[1, "review_score"] == 3
    assert result.loc[2, "review_score"] == 5
    assert pd.isna(result.loc[3, "review_score"])
    assert pd.isna(result.loc[4, "review_score"])
    assert pd.isna(result.loc[5, "review_score"])

def test_drop_high_null_rows():
    df = pd.DataFrame({
        "a": [1, None, None, 4],
        "b": [2, None, 3, 5],
        "c": [3, None, None, 6],
        "d": [4, None, 5, 7],
    })
    result = _drop_high_null_rows(
        df,
        "test_table",
        threshold=0.5,
    )
    # Row 1 has 4/4 NULL values and must be removed.
    # Row 2 has 2/4 NULL values and is kept.
    assert len(result) == 3

def test_drop_orphan_fk():
    df = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "value": ["A", "B", "C"],
    })
    valid_keys = {
        "customers": {1, 3},
    }
    result = _drop_orphan_fk(
        df,
        "orders",
        "customer_id",
        valid_keys,
        "customers",
    )
    assert result["customer_id"].tolist() == [1, 3]

def test_clean_customers():
    df = pd.DataFrame({
        "customer_id": [1, 1, None, 2],
        "name": [" Alice ", "Alice", "Bob", " Carol "],
    })
    result = clean_dataframe("customers", df)
    assert result["customer_id"].tolist() == [1, 2]
    assert result["name"].tolist() == ["Alice", "Carol"]

def test_clean_sellers():
    df = pd.DataFrame({
        "seller_id": [1, 1, None, 2],
        "name": [" Seller A ", "Seller A", "Seller B", " Seller C "],
    })
    result = clean_dataframe("sellers", df)
    assert result["seller_id"].tolist() == [1, 2]
    assert result["name"].tolist() == ["Seller A", "Seller C"]

def test_clean_products():
    df = pd.DataFrame({
        "product_id": ["p1", "p2", "p3"],
        "product_weight_g": ["100", "-20", "invalid"],
        "product_length_cm": ["10", "20", "30"],
        "product_height_cm": ["5", "6", "7"],
        "product_width_cm": ["2", "3", "4"],
    })
    result = clean_dataframe("products", df)
    assert result["product_id"].tolist() == ["p1", "p3"]
    assert result.loc[result["product_id"] == "p1", "product_weight_g"].iloc[0] == 100
    assert pd.isna(
        result.loc[result["product_id"] == "p3", "product_weight_g"].iloc[0]
    )

def test_clean_orders():
    df = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "customer_id": ["c1", "c2", "c3"],
        "order_status": ["delivered", "shipped", "canceled"],
        "order_purchase_timestamp": ["2026-01-01", "2026-01-02", "2026-01-03"],
    })
    valid_keys = {
        "customers": {"c1", "c3"},
    }
    result = clean_dataframe(
        "orders",
        df,
        valid_keys=valid_keys,
    )
    assert result["order_id"].tolist() == ["o1", "o3"]

def test_clean_order_items():
    df = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "product_id": ["p1", "p2", "p3"],
        "seller_id": ["s1", "s2", "s3"],
        "price": ["100", "-20", "50"],
        "freight_value": ["10", "5", "-2"],
    })
    valid_keys = {
        "orders": {"o1", "o2", "o3"},
        "products": {"p1", "p2", "p3"},
        "sellers": {"s1", "s2", "s3"},
    }
    result = clean_dataframe(
        "order_items",
        df,
        valid_keys=valid_keys,
    )
    assert result["order_id"].tolist() == ["o1"]

def test_clean_order_payments():
    df = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "payment_value": ["100", "-20", "50"],
    })
    valid_keys = {
        "orders": {"o1", "o3"},
    }
    result = clean_dataframe(
        "order_payments",
        df,
        valid_keys=valid_keys,
    )
    assert result["order_id"].tolist() == ["o1", "o3"]
    assert result["payment_value"].tolist() == [100, 50]

def test_clean_order_reviews():
    df = pd.DataFrame({
        "review_id": ["r1", "r2", "r3", "r4"],
        "order_id": ["o1", "o2", "o3", "o4"],
        "review_score": ["5", "3", "0", "6"],
        "review_creation_date": [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-04",
        ],
        "review_comment_title": [None, " Good ", None, None],
        "review_comment_message": [None, " Nice ", None, None],
    })
    valid_keys = {
        "orders": {"o1", "o2", "o3", "o4"},
    }
    result = clean_dataframe(
        "order_reviews",
        df,
        valid_keys=valid_keys,
    )
    assert result["review_id"].tolist() == ["r1", "r2"]
    assert result["review_score"].tolist() == [5, 3]
    assert result.loc[result["review_id"] == "r1", "review_comment_title"].iloc[0] == ""
    assert result.loc[result["review_id"] == "r1", "review_comment_message"].iloc[0] == ""

def test_clean_geolocation():
    df = pd.DataFrame({
        "geolocation_zip_code_prefix": [100, 200, 300],
        "geolocation_lat": [17.0, 100.0, 20.0],
        "geolocation_lng": [78.0, 80.0, 200.0],
    })
    result = clean_dataframe("geolocation", df)
    # Rows with invalid/out-of-range coordinates are dropped
    assert len(result) == 1
    assert result.iloc[0]["geolocation_zip_code_prefix"] == 100
    assert result.iloc[0]["geolocation_lat"] == 17.0
    assert result.iloc[0]["geolocation_lng"] == 78.0

