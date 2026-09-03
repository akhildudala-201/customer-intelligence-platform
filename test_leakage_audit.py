"""Pytest checks for the combined customer feature and churn-label dataset."""

from pathlib import Path

import pandas as pd
import pytest
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "outputs" / "customer_features_with_labels.csv"
CHURN_THRESHOLD_DAYS = 180

REQUIRED_COLUMNS = {
    "customer_unique_id",
    "recency_days",
    "last_purchase_date",
    "reference_date",
    "churn_label",
    "censored",
}


def audit_dataset(dataset: pd.DataFrame) -> list[str]:
    """Return all validation errors found in a combined dataset."""
    errors: list[str] = []

    missing_columns = sorted(REQUIRED_COLUMNS - set(dataset.columns))
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return errors

    customer_ids = dataset["customer_unique_id"]
    if customer_ids.isna().any() or customer_ids.astype(str).str.strip().eq("").any():
        errors.append("customer_unique_id contains null or empty values.")
    if customer_ids.duplicated().any():
        errors.append(
            f"customer_unique_id contains {int(customer_ids.duplicated().sum())} duplicates."
        )

    labels = dataset["churn_label"]
    invalid_labels = ~labels.isin([0, 1])
    if invalid_labels.any():
        errors.append(
            f"churn_label contains {int(invalid_labels.sum())} values other than 0 or 1."
        )

    censored = dataset["censored"]
    invalid_censored = ~censored.isin([0, 1])
    if invalid_censored.any():
        errors.append(
            f"censored contains {int(invalid_censored.sum())} values other than 0 or 1."
        )

    numeric_columns = dataset.select_dtypes(include="number").columns
    for column in numeric_columns:
        if not np.isfinite(dataset[column].dropna()).all():
            errors.append(f"{column} contains non-finite numeric values.")

    for column in ("last_purchase_date", "reference_date"):
        parsed_dates = pd.to_datetime(dataset[column], errors="coerce")
        if parsed_dates.isna().any():
            errors.append(f"{column} contains invalid dates.")

    recency = pd.to_numeric(dataset["recency_days"], errors="coerce")
    if recency.isna().any() or (recency < 0).any():
        errors.append("recency_days contains invalid or negative values.")
    if not errors:
        expected_labels = pd.Series(1, index=dataset.index)
        mismatches = (labels.astype(int) != expected_labels).sum()
        if mismatches:
            errors.append(
                f"churn_label must be 1 for all included rows; "
                f"{int(mismatches)} rows disagree."
            )
        threshold = CHURN_THRESHOLD_DAYS
        expected_censored = (recency <= threshold).astype(int)
        censored_mismatches = (
            censored.astype(int) != expected_censored
        ).sum()
        if censored_mismatches:
            errors.append(
                f"censored disagrees with the {CHURN_THRESHOLD_DAYS}-day "
                f"recency rule for {int(censored_mismatches)} rows."
            )

    return errors


@pytest.fixture
def valid_dataset():
    return pd.DataFrame(
        {
            "customer_unique_id": ["u1", "u2", "u3"],
            "recency_days": [200, 220, 300],
            "last_purchase_date": ["2018-10-17"] * 3,
            "reference_date": ["2018-10-17"] * 3,
            "churn_label": [1, 1, 1],
            "censored": [0, 0, 0],
        }
    )


def test_combined_dataset_passes_audit():
    if not DATASET_PATH.exists():
        pytest.skip(f"Combined dataset not found: {DATASET_PATH}")

    dataset = pd.read_csv(DATASET_PATH)
    assert audit_dataset(dataset) == []


def test_audit_detects_duplicate_customer_ids(valid_dataset):
    invalid = pd.concat([valid_dataset, valid_dataset.iloc[[0]]], ignore_index=True)
    errors = audit_dataset(invalid)
    assert any("duplicate" in error for error in errors)


def test_audit_detects_missing_required_columns(valid_dataset):
    invalid = valid_dataset.drop(columns=["reference_date"])
    errors = audit_dataset(invalid)
    assert any("Missing required columns" in error for error in errors)


@pytest.mark.parametrize("bad_id", [None, ""])
def test_audit_detects_missing_customer_ids(valid_dataset, bad_id):
    invalid = valid_dataset.copy()
    invalid.loc[0, "customer_unique_id"] = bad_id
    errors = audit_dataset(invalid)
    assert any("null or empty" in error for error in errors)


def test_audit_detects_invalid_labels(valid_dataset):
    invalid = valid_dataset.copy()
    invalid.loc[0, "churn_label"] = 2
    errors = audit_dataset(invalid)
    assert any("other than 0 or 1" in error for error in errors)


def test_audit_detects_non_finite_numeric_values(valid_dataset):
    invalid = valid_dataset.copy()
    invalid["recency_days"] = [10, float("inf"), 30]
    errors = audit_dataset(invalid)
    assert any("non-finite" in error for error in errors)


@pytest.mark.parametrize("bad_date", ["not-a-date", None])
def test_audit_detects_invalid_dates(valid_dataset, bad_date):
    invalid = valid_dataset.copy()
    invalid.loc[0, "last_purchase_date"] = bad_date
    errors = audit_dataset(invalid)
    assert any("last_purchase_date" in error for error in errors)


def test_audit_detects_negative_recency(valid_dataset):
    invalid = valid_dataset.copy()
    invalid.loc[0, "recency_days"] = -1
    errors = audit_dataset(invalid)
    assert any("negative" in error for error in errors)


def test_audit_detects_non_numeric_recency(valid_dataset):
    invalid = valid_dataset.copy()
    invalid["recency_days"] = invalid["recency_days"].astype(object)
    invalid.loc[0, "recency_days"] = "unknown"
    errors = audit_dataset(invalid)
    assert any("invalid or negative" in error for error in errors)


def test_audit_detects_label_mismatch(valid_dataset):
    invalid = valid_dataset.copy()
    invalid.loc[0, "churn_label"] = 0
    errors = audit_dataset(invalid)
    assert any("must be 1" in error for error in errors)


def test_audit_detects_censored_value_mismatch(valid_dataset):
    invalid = valid_dataset.copy()
    invalid.loc[0, "recency_days"] = 180
    invalid.loc[0, "censored"] = 0
    errors = audit_dataset(invalid)
    assert any("censored disagrees" in error for error in errors)
