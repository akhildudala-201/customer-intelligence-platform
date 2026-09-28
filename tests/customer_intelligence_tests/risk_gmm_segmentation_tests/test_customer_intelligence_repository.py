import pandas as pd
import pytest

from app.segmentation.customer_intelligence.data_access.customer_intelligence_repository import (
    validate_merged_dataset,
)


def make_valid_dataframe():
    return pd.DataFrame(
        {
            "customer_unique_id": ["C001", "C002", "C003"],
            "churn_probability": [0.20, 0.55, 0.85],
        }
    )


def test_validate_merged_dataset_accepts_valid_data():
    df = make_valid_dataframe()

    validate_merged_dataset(df)


def test_validate_merged_dataset_rejects_empty_dataframe():
    df = pd.DataFrame(
        columns=[
            "customer_unique_id",
            "churn_probability",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Merged customer_intelligence base is empty",
    ):
        validate_merged_dataset(df)


def test_validate_merged_dataset_rejects_duplicate_customer_ids():
    df = pd.DataFrame(
        {
            "customer_unique_id": [
                "C001",
                "C001",
                "C002",
            ],
            "churn_probability": [
                0.20,
                0.30,
                0.80,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Duplicate customer_unique_id",
    ):
        validate_merged_dataset(df)


def test_validate_merged_dataset_rejects_null_churn_probability():
    df = pd.DataFrame(
        {
            "customer_unique_id": [
                "C001",
                "C002",
            ],
            "churn_probability": [
                0.20,
                None,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="null churn_probability",
    ):
        validate_merged_dataset(df)


@pytest.mark.parametrize(
    "invalid_probability",
    [
        -0.01,
        1.01,
        2.0,
    ],
)
def test_validate_merged_dataset_rejects_out_of_range_probability(
    invalid_probability,
):
    df = pd.DataFrame(
        {
            "customer_unique_id": ["C001"],
            "churn_probability": [invalid_probability],
        }
    )

    with pytest.raises(
        ValueError,
        match="outside \\[0, 1\\]",
    ):
        validate_merged_dataset(df)