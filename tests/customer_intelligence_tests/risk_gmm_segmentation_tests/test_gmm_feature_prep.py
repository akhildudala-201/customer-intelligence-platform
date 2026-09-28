import numpy as np
import pandas as pd
import pytest

from app.segmentation.customer_intelligence.gmm.feature_prep import (
    BEHAVIOURAL_FEATURES,
    CHURN_PROBABILITY_COLUMN,
    CHURN_PROBABILITY_WEIGHT,
    RISK_TIER_WEIGHT,
    scale_features,
    select_segmentation_features,
)


def make_customer_data():
    return pd.DataFrame(
        {
            "customer_unique_id": [
                "C001",
                "C002",
                "C003",
                "C004",
            ],
            "monetary_value": [
                100.0,
                200.0,
                300.0,
                400.0,
            ],
            "frequency": [
                1,
                2,
                3,
                4,
            ],
            "recency_days": [
                100,
                80,
                40,
                10,
            ],
            "avg_review_score": [
                3.0,
                4.0,
                4.5,
                5.0,
            ],
            "avg_delivery_days": [
                10.0,
                8.0,
                6.0,
                5.0,
            ],
            "avg_delivery_delay_days": [
                3.0,
                2.0,
                1.0,
                0.0,
            ],
            "avg_payment_installments": [
                2.0,
                3.0,
                2.0,
                1.0,
            ],
            "freight_ratio": [
                0.10,
                0.20,
                0.15,
                0.12,
            ],
            "has_bad_review": [
                1,
                0,
                0,
                0,
            ],
            "has_review_comment": [
                1,
                1,
                0,
                0,
            ],
            "is_delayed_delivery": [
                1,
                1,
                0,
                0,
            ],
            "churn_probability": [
                0.10,
                0.30,
                0.60,
                0.90,
            ],
        }
    )


def make_risk_tier_data():
    return pd.DataFrame(
        {
            "customer_unique_id": [
                "C001",
                "C002",
                "C003",
                "C004",
            ],
            "risk_tier": [
                "Low Risk",
                "Low Risk",
                "Medium Risk",
                "High Risk",
            ],
        }
    )


def test_select_segmentation_features():
    customer_df = make_customer_data()
    risk_df = make_risk_tier_data()

    result = select_segmentation_features(
        customer_df,
        risk_df,
    )

    assert len(result) == 4

    assert "customer_unique_id" in result.columns
    assert "risk_tier" in result.columns

    for column in BEHAVIOURAL_FEATURES:
        assert column in result.columns

    assert CHURN_PROBABILITY_COLUMN in result.columns


def test_select_segmentation_features_excludes_missing_rows():
    customer_df = make_customer_data()
    risk_df = make_risk_tier_data()

    customer_df.loc[1, "frequency"] = np.nan

    result = select_segmentation_features(
        customer_df,
        risk_df,
    )

    assert len(result) == 3
    assert "C002" not in result["customer_unique_id"].tolist()


def test_select_segmentation_features_rejects_missing_feature_column():
    customer_df = make_customer_data()
    risk_df = make_risk_tier_data()

    customer_df = customer_df.drop(
        columns=["frequency"]
    )

    with pytest.raises(
        KeyError,
        match="frequency",
    ):
        select_segmentation_features(
            customer_df,
            risk_df,
        )


def test_select_segmentation_features_requires_one_to_one_risk_tiers():
    customer_df = make_customer_data()
    risk_df = make_risk_tier_data()

    duplicate_row = pd.DataFrame(
        {
            "customer_unique_id": ["C001"],
            "risk_tier": ["Low Risk"],
        }
    )

    risk_df = pd.concat(
        [
            risk_df,
            duplicate_row,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
    ):
        select_segmentation_features(
            customer_df,
            risk_df,
        )


def test_scale_features_returns_scaler():
    customer_df = make_customer_data()
    risk_df = make_risk_tier_data()

    scoreable_df = select_segmentation_features(
        customer_df,
        risk_df,
    )

    scaled_df, scaler = scale_features(
        scoreable_df,
    )

    assert scaler is not None
    assert len(scaled_df) == len(scoreable_df)


def test_scale_features_contains_expected_columns():
    customer_df = make_customer_data()
    risk_df = make_risk_tier_data()

    scoreable_df = select_segmentation_features(
        customer_df,
        risk_df,
    )

    scaled_df, _ = scale_features(
        scoreable_df,
    )

    for column in BEHAVIOURAL_FEATURES:
        assert column in scaled_df.columns

    assert "churn_probability" in scaled_df.columns

    assert "risk_tier__Low Risk" in scaled_df.columns
    assert "risk_tier__Medium Risk" in scaled_df.columns
    assert "risk_tier__High Risk" in scaled_df.columns


def test_scale_features_applies_churn_probability_weight():
    customer_df = make_customer_data()
    risk_df = make_risk_tier_data()

    scoreable_df = select_segmentation_features(
        customer_df,
        risk_df,
    )

    scaled_df, scaler = scale_features(
        scoreable_df,
    )

    raw_scaled = scaler.transform(
        scoreable_df[
            BEHAVIOURAL_FEATURES
            + ["churn_probability"]
        ]
    )

    churn_index = len(BEHAVIOURAL_FEATURES)

    expected = (
        raw_scaled[:, churn_index]
        * CHURN_PROBABILITY_WEIGHT
    )

    np.testing.assert_allclose(
        scaled_df["churn_probability"].values,
        expected,
    )


def test_scale_features_applies_risk_tier_weight():
    customer_df = make_customer_data()
    risk_df = make_risk_tier_data()

    scoreable_df = select_segmentation_features(
        customer_df,
        risk_df,
    )

    scaled_df, _ = scale_features(
        scoreable_df,
    )

    for tier in [
        "Low Risk",
        "Medium Risk",
        "High Risk",
    ]:
        column = f"risk_tier__{tier}"

        assert set(
            scaled_df[column].unique()
        ).issubset(
            {0.0, RISK_TIER_WEIGHT}
        )