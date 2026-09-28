import pandas as pd
import pytest

from app.segmentation.customer_intelligence.gmm.labeling import (
    profile_clusters,
    profile_risk_tier_mix,
    suggest_segment_labels,
)


def make_scoreable_data():
    return pd.DataFrame(
        {
            "monetary_value": [
                100,
                120,
                500,
                550,
            ],
            "frequency": [
                1,
                1,
                3,
                4,
            ],
            "recency_days": [
                100,
                90,
                20,
                10,
            ],
            "avg_review_score": [
                2.0,
                2.5,
                4.5,
                4.8,
            ],
            "avg_delivery_days": [
                12,
                11,
                5,
                5,
            ],
            "avg_delivery_delay_days": [
                3,
                2,
                0,
                0,
            ],
            "avg_payment_installments": [
                2,
                2,
                3,
                3,
            ],
            "freight_ratio": [
                0.2,
                0.2,
                0.1,
                0.1,
            ],
            "has_bad_review": [
                1,
                1,
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
                0.8,
                0.7,
                0.2,
                0.1,
            ],
            "risk_tier": [
                "High Risk",
                "High Risk",
                "Low Risk",
                "Low Risk",
            ],
        }
    )


def make_cluster_assignments():
    return pd.DataFrame(
        {
            "segment_id": [
                0,
                0,
                1,
                1,
            ],
            "cluster_probability": [
                0.90,
                0.85,
                0.95,
                0.92,
            ],
        }
    )


def test_profile_clusters():
    scoreable_df = make_scoreable_data()
    assignments = make_cluster_assignments()

    result = profile_clusters(
        scoreable_df,
        assignments,
        [
            "monetary_value",
            "frequency",
            "avg_review_score",
        ],
    )

    assert len(result) == 2

    assert "segment_id" in result.columns
    assert "customer_count" in result.columns
    assert "monetary_value" in result.columns
    assert "frequency" in result.columns


def test_profile_clusters_calculates_customer_count():
    scoreable_df = make_scoreable_data()
    assignments = make_cluster_assignments()

    result = profile_clusters(
        scoreable_df,
        assignments,
        ["monetary_value"],
    )

    counts = (
        result
        .set_index("segment_id")[
            "customer_count"
        ]
        .to_dict()
    )

    assert counts[0] == 2
    assert counts[1] == 2


def test_profile_risk_tier_mix():
    scoreable_df = make_scoreable_data()
    assignments = make_cluster_assignments()

    result = profile_risk_tier_mix(
        scoreable_df,
        assignments,
        "risk_tier",
    )

    assert len(result) == 2

    assert "segment_id" in result.columns
    assert "share_High Risk" in result.columns
    assert "share_Low Risk" in result.columns


def test_profile_risk_tier_mix_percentages_sum_to_one():
    scoreable_df = make_scoreable_data()
    assignments = make_cluster_assignments()

    result = profile_risk_tier_mix(
        scoreable_df,
        assignments,
        "risk_tier",
    )

    share_columns = [
        column
        for column in result.columns
        if column.startswith("share_")
    ]

    totals = result[share_columns].sum(axis=1)

    assert (totals == 1.0).all()


def test_suggest_segment_labels_returns_expected_columns():
    cluster_profile = pd.DataFrame(
        {
            "segment_id": [0, 1],
            "monetary_value": [
                100,
                500,
            ],
            "frequency": [
                1.0,
                4.0,
            ],
            "avg_review_score": [
                2.0,
                4.8,
            ],
            "avg_delivery_days": [
                12.0,
                5.0,
            ],
            "avg_delivery_delay_days": [
                3.0,
                0.0,
            ],
            "has_bad_review": [
                1.0,
                0.0,
            ],
            "is_delayed_delivery": [
                1.0,
                0.0,
            ],
            "has_review_comment": [
                1.0,
                0.0,
            ],
        }
    )

    result = suggest_segment_labels(
        cluster_profile
    )

    assert list(result.columns) == [
        "segment_id",
        "segment_label",
    ]

    assert len(result) == 2


def test_suggest_segment_labels_identifies_repeat_buyers():
    cluster_profile = pd.DataFrame(
        {
            "segment_id": [0, 1],
            "monetary_value": [
                100,
                100,
            ],
            "frequency": [
                1.0,
                3.0,
            ],
            "avg_review_score": [
                4.0,
                4.0,
            ],
            "avg_delivery_days": [
                5.0,
                5.0,
            ],
            "avg_delivery_delay_days": [
                0.0,
                0.0,
            ],
            "has_bad_review": [
                0.0,
                0.0,
            ],
            "is_delayed_delivery": [
                0.0,
                0.0,
            ],
            "has_review_comment": [
                0.0,
                0.0,
            ],
        }
    )

    result = suggest_segment_labels(
        cluster_profile
    )

    assert (
        "One-Time Buyers"
        in result.loc[0, "segment_label"]
    )

    assert (
        "Repeat Buyers"
        in result.loc[1, "segment_label"]
    )


def test_suggest_segment_labels_rejects_missing_columns():
    cluster_profile = pd.DataFrame(
        {
            "segment_id": [0],
            "frequency": [2.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing column",
    ):
        suggest_segment_labels(
            cluster_profile
        )