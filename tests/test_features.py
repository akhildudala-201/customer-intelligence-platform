import numpy as np
import pandas as pd
import pytest

from app.Features.encoding_transformation import (
    frequency_encode,
    one_hot_encode,
    apply_log1p,
    transform_data,
    validate_output,
)
from app.Features.feature_selection_and_scaling import (
    drop_censored,
    time_split_data,
    select_features,
    fit_scaler,
    apply_scaler,
    impute_missing,
)


def test_frequency_encode():
    df = pd.DataFrame({
        "dominant_product_category": ["catA", "catB", "catA", "catA"],
        "customer_city_state": ["NY, NY", "CA, CA", "NY, NY", "TX, TX"],
    })
    encoded = frequency_encode(df)
    assert "dominant_product_category" not in encoded.columns
    assert "dominant_product_category_frequency" in encoded.columns
    assert encoded["dominant_product_category_frequency"].iloc[0] == 0.75
    assert encoded["dominant_product_category_frequency"].iloc[1] == 0.25


def test_one_hot_encode():
    df = pd.DataFrame({
        "latest_order_status": ["delivered", "canceled", "shipped"],
        "preferred_payment_type": ["credit_card", "boleto", "credit_card"],
    })
    encoded = one_hot_encode(df)
    assert "latest_order_status" not in encoded.columns
    assert "preferred_payment_type" not in encoded.columns
    assert "latest_order_status_delivered" in encoded.columns
    assert "preferred_payment_type_credit_card" in encoded.columns
    assert encoded["latest_order_status_delivered"].tolist() == [1, 0, 0]


def test_apply_log1p():
    df = pd.DataFrame({
        "recency_days": [0.0, 10.0, 99.0],
        "monetary_value": [0.0, 100.0, 500.0],
    })
    transformed = apply_log1p(df)
    assert np.isclose(transformed["recency_days"].iloc[0], 0.0)
    assert np.isclose(transformed["recency_days"].iloc[1], np.log1p(10.0))


def test_apply_log1p_negative_raises():
    df = pd.DataFrame({
        "recency_days": [-5.0, 10.0],
    })
    with pytest.raises(ValueError, match="Negative values found"):
        apply_log1p(df)


def test_drop_censored(sample_features_with_labels_df):
    uncensored = drop_censored(sample_features_with_labels_df)
    assert uncensored["churn_label"].notna().all()
    assert len(uncensored) <= len(sample_features_with_labels_df)


def test_time_split_data(sample_features_with_labels_df):
    uncensored = drop_censored(sample_features_with_labels_df)
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     ids_train, ids_val, ids_test) = time_split_data(uncensored)

    assert len(X_train) + len(X_val) + len(X_test) == len(uncensored)
    # Check that there is no ID leakage
    set_train = set(ids_train)
    set_val = set(ids_val)
    set_test = set(ids_test)
    assert len(set_train & set_val) == 0
    assert len(set_train & set_test) == 0
    assert len(set_val & set_test) == 0


def test_scaling():
    X_train = pd.DataFrame({
        "f1": [10.0, 20.0, 30.0, 40.0, 50.0],
        "f2": [100.0, 200.0, 300.0, 400.0, 500.0],
    })
    scaler = fit_scaler(X_train)
    scaled = apply_scaler(X_train, scaler)
    assert scaled.shape == X_train.shape
    assert np.isclose(scaled["f1"].median(), 0.0)


def test_feature_selection():
    np.random.seed(42)
    n = 200
    y = pd.Series(np.random.choice([0, 1], size=n))
    # Highly correlated feature pair
    f1 = np.random.randn(n)
    f2 = f1 * 0.999 + np.random.randn(n) * 0.001
    f3 = np.random.randn(n)
    binary_sig = y.copy()  # perfect correlation with target
    binary_noise = pd.Series(np.random.choice([0, 1], size=n))

    X = pd.DataFrame({
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "bin_sig": binary_sig,
        "bin_noise": binary_noise,
    })

    kept = select_features(X, y, corr_threshold=0.90)
    # One of f1 or f2 must be dropped
    assert not ("f1" in kept and "f2" in kept)
    assert "bin_sig" in kept
