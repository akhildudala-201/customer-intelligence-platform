from pathlib import Path

import pandas as pd
import pytest

from app.segmentation.customer_intelligence.risk_tier.classifier import (
    classify_risk_tier,
    classify_risk_tiers,
    load_risk_tier_config,
)


@pytest.fixture(autouse=True)
def clear_config_cache():
    load_risk_tier_config.cache_clear()

    yield

    load_risk_tier_config.cache_clear()


def test_load_risk_tier_config():
    tiers = load_risk_tier_config()

    assert len(tiers) == 3

    assert tiers[0] == (0.39, "Low Risk")
    assert tiers[1] == (0.69, "Medium Risk")
    assert tiers[2] == (1.00, "High Risk")


@pytest.mark.parametrize(
    "probability, expected_tier",
    [
        (0.00, "Low Risk"),
        (0.20, "Low Risk"),
        (0.39, "Low Risk"),
        (0.40, "Medium Risk"),
        (0.55, "Medium Risk"),
        (0.69, "Medium Risk"),
        (0.70, "High Risk"),
        (0.85, "High Risk"),
        (1.00, "High Risk"),
    ],
)
def test_classify_risk_tier(
    probability,
    expected_tier,
):
    assert classify_risk_tier(probability) == expected_tier


def test_classify_risk_tiers():
    df = pd.DataFrame(
        {
            "customer_unique_id": [
                "C001",
                "C002",
                "C003",
            ],
            "churn_probability": [
                0.20,
                0.55,
                0.85,
            ],
        }
    )

    result = classify_risk_tiers(df)

    assert list(result.columns) == [
        "customer_unique_id",
        "churn_probability",
        "risk_tier",
    ]

    assert result["risk_tier"].tolist() == [
        "Low Risk",
        "Medium Risk",
        "High Risk",
    ]


def test_classify_risk_tiers_preserves_customer_ids():
    df = pd.DataFrame(
        {
            "customer_unique_id": [
                "C001",
                "C002",
            ],
            "churn_probability": [
                0.10,
                0.90,
            ],
        }
    )

    result = classify_risk_tiers(df)

    assert result["customer_unique_id"].tolist() == [
        "C001",
        "C002",
    ]


def test_risk_config_rejects_missing_tiers(tmp_path):
    config_file = tmp_path / "empty.yaml"

    config_file.write_text(
        "tiers: []",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="No tiers defined",
    ):
        load_risk_tier_config(config_file)


def test_risk_config_rejects_config_not_reaching_one(tmp_path):
    config_file = tmp_path / "invalid.yaml"

    config_file.write_text(
        """
tiers:
  - name: Low Risk
    upper_bound: 0.50
  - name: High Risk
    upper_bound: 0.90
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="expected 1.0",
    ):
        load_risk_tier_config(config_file)


def test_risk_config_rejects_overlapping_tiers(tmp_path):
    config_file = tmp_path / "invalid.yaml"

    config_file.write_text(
        """
tiers:
  - name: Low Risk
    upper_bound: 0.60
  - name: Medium Risk
    upper_bound: 0.50
  - name: High Risk
    upper_bound: 1.00
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must be greater",
    ):
        load_risk_tier_config(config_file)