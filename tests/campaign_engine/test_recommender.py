import pandas as pd
import pytest

from app.segmentation.customer_intelligence.campaign_engine.recommender import (
    recommend_campaigns,
)

from app.segmentation.customer_intelligence.campaign_engine.campaign_rules_loader import (
    CampaignRule,
    CampaignRulesConfig,
)


@pytest.fixture
def campaign_config():
    """
    Small in-memory configuration used for unit testing.

    This avoids reading the YAML file and makes the tests independent
    of the current campaign configuration on disk.
    """

    return CampaignRulesConfig(
        risk_tiers=(
            "High Risk",
            "Medium Risk",
            "Low Risk",
        ),
        segments=(
            "Satisfied One-Time Buyers",
            "High-Value Satisfied Repeat Buyers",
            "Product-Dissatisfied One-Time Buyers",
        ),
        value_tiers=(
            "High",
            "Medium",
            "Low",
        ),
        reason_codes={
            "RC00": "General Engagement",
            "RC01": "Priority Retention",
            "RC02": "Quality Recovery",
            "RC03": "Loyalty Reward",
            "RC04": "Repeat Purchase Prompt",
        },
        default_rule=CampaignRule(
            campaign_name="Standard Customer Engagement Program",
            campaign_priority=5,
            reason_code="RC00",
        ),
        rules={
            (
                "High Risk",
                "Satisfied One-Time Buyers",
                "High",
            ): CampaignRule(
                campaign_name="Repeat Purchase Incentive Program",
                campaign_priority=2,
                reason_code="RC04",
            ),
            (
                "High Risk",
                "Satisfied One-Time Buyers",
                "Medium",
            ): CampaignRule(
                campaign_name="Second Purchase Reminder Campaign",
                campaign_priority=3,
                reason_code="RC04",
            ),
            (
                "High Risk",
                "Satisfied One-Time Buyers",
                "Low",
            ): CampaignRule(
                campaign_name="Second Purchase SMS Campaign",
                campaign_priority=4,
                reason_code="RC04",
            ),
            (
                "High Risk",
                "High-Value Satisfied Repeat Buyers",
                "High",
            ): CampaignRule(
                campaign_name="Executive Retention Outreach",
                campaign_priority=1,
                reason_code="RC01",
            ),
            (
                "High Risk",
                "Product-Dissatisfied One-Time Buyers",
                "High",
            ): CampaignRule(
                campaign_name="Priority Service Recovery Call",
                campaign_priority=1,
                reason_code="RC02",
            ),
        },
    )


def test_recommend_campaigns_matches_high_value_satisfied_customer(
    campaign_config,
):
    """Test an explicit campaign rule."""
    df = pd.DataFrame(
        [
            {
                "customer_unique_id": "customer_001",
                "risk_tier": "High Risk",
                "segment_label": "Satisfied One-Time Buyers",
                "value_tier": "High",
            }
        ]
    )

    result = recommend_campaigns(df, config=campaign_config)

    assert len(result) == 1
    assert result.loc[0, "campaign_name"] == (
        "Repeat Purchase Incentive Program"
    )
    assert result.loc[0, "campaign_priority"] == 2
    assert result.loc[0, "reason_code"] == "RC04"


def test_recommend_campaigns_matches_medium_value_customer(
    campaign_config,
):
    """Test another explicit rule."""
    df = pd.DataFrame(
        [
            {
                "customer_unique_id": "customer_002",
                "risk_tier": "High Risk",
                "segment_label": "Satisfied One-Time Buyers",
                "value_tier": "Medium",
            }
        ]
    )

    result = recommend_campaigns(df, config=campaign_config)

    assert result.loc[0, "campaign_name"] == (
        "Second Purchase Reminder Campaign"
    )
    assert result.loc[0, "campaign_priority"] == 3
    assert result.loc[0, "reason_code"] == "RC04"


def test_recommend_campaigns_matches_low_value_customer(
    campaign_config,
):
    """Test the low-value explicit rule."""
    df = pd.DataFrame(
        [
            {
                "customer_unique_id": "customer_003",
                "risk_tier": "High Risk",
                "segment_label": "Satisfied One-Time Buyers",
                "value_tier": "Low",
            }
        ]
    )

    result = recommend_campaigns(df, config=campaign_config)

    assert result.loc[0, "campaign_name"] == (
        "Second Purchase SMS Campaign"
    )
    assert result.loc[0, "campaign_priority"] == 4
    assert result.loc[0, "reason_code"] == "RC04"


def test_recommend_campaigns_matches_repeat_buyer_rule(
    campaign_config,
):
    """Test a different segment."""
    df = pd.DataFrame(
        [
            {
                "customer_unique_id": "customer_004",
                "risk_tier": "High Risk",
                "segment_label": "High-Value Satisfied Repeat Buyers",
                "value_tier": "High",
            }
        ]
    )

    result = recommend_campaigns(df, config=campaign_config)

    assert result.loc[0, "campaign_name"] == (
        "Executive Retention Outreach"
    )
    assert result.loc[0, "campaign_priority"] == 1
    assert result.loc[0, "reason_code"] == "RC01"


def test_recommend_campaigns_matches_dissatisfied_customer(
    campaign_config,
):
    """Test the product-dissatisfaction rule."""
    df = pd.DataFrame(
        [
            {
                "customer_unique_id": "customer_005",
                "risk_tier": "High Risk",
                "segment_label": "Product-Dissatisfied One-Time Buyers",
                "value_tier": "High",
            }
        ]
    )

    result = recommend_campaigns(df, config=campaign_config)

    assert result.loc[0, "campaign_name"] == (
        "Priority Service Recovery Call"
    )
    assert result.loc[0, "campaign_priority"] == 1
    assert result.loc[0, "reason_code"] == "RC02"


def test_unmatched_combination_uses_default_rule(campaign_config):
    """
    An unknown combination should fall back to default_rule.
    """
    df = pd.DataFrame(
        [
            {
                "customer_unique_id": "customer_006",
                "risk_tier": "Low Risk",
                "segment_label": "Satisfied One-Time Buyers",
                "value_tier": "Low",
            }
        ]
    )

    result = recommend_campaigns(df, config=campaign_config)

    assert result.loc[0, "campaign_name"] == (
        "Standard Customer Engagement Program"
    )
    assert result.loc[0, "campaign_priority"] == 5
    assert result.loc[0, "reason_code"] == "RC00"


def test_unknown_segment_uses_default_rule(campaign_config, capsys):
    """
    This is especially useful for your current GMM situation.

    If GMM produces a segment that is not declared in the campaign
    configuration, the recommender should warn and use default_rule.
    """
    df = pd.DataFrame(
        [
            {
                "customer_unique_id": "customer_007",
                "risk_tier": "High Risk",
                "segment_label": "Unknown GMM Segment",
                "value_tier": "High",
            }
        ]
    )

    result = recommend_campaigns(df, config=campaign_config)

    captured = capsys.readouterr()

    assert "segment_label" in captured.out
    assert "Unknown GMM Segment" in captured.out

    assert result.loc[0, "campaign_name"] == (
        "Standard Customer Engagement Program"
    )
    assert result.loc[0, "campaign_priority"] == 5
    assert result.loc[0, "reason_code"] == "RC00"


def test_multiple_customers_are_recommended(campaign_config):
    """Verify multiple customers can be processed in one call."""
    df = pd.DataFrame(
        [
            {
                "customer_unique_id": "customer_001",
                "risk_tier": "High Risk",
                "segment_label": "Satisfied One-Time Buyers",
                "value_tier": "High",
            },
            {
                "customer_unique_id": "customer_002",
                "risk_tier": "High Risk",
                "segment_label": "Satisfied One-Time Buyers",
                "value_tier": "Medium",
            },
            {
                "customer_unique_id": "customer_003",
                "risk_tier": "High Risk",
                "segment_label": "Satisfied One-Time Buyers",
                "value_tier": "Low",
            },
        ]
    )

    result = recommend_campaigns(df, config=campaign_config)

    assert len(result) == 3

    assert result.loc[0, "campaign_name"] == (
        "Repeat Purchase Incentive Program"
    )
    assert result.loc[1, "campaign_name"] == (
        "Second Purchase Reminder Campaign"
    )
    assert result.loc[2, "campaign_name"] == (
        "Second Purchase SMS Campaign"
    )


def test_output_columns_are_correct(campaign_config):
    """Verify the schema-6.3 output columns."""
    df = pd.DataFrame(
        [
            {
                "customer_unique_id": "customer_008",
                "risk_tier": "High Risk",
                "segment_label": "Satisfied One-Time Buyers",
                "value_tier": "High",
            }
        ]
    )

    result = recommend_campaigns(df, config=campaign_config)

    expected_columns = [
        "customer_unique_id",
        "risk_tier",
        "segment_label",
        "value_tier",
        "campaign_name",
        "campaign_priority",
        "reason_code",
    ]

    assert list(result.columns) == expected_columns


def test_missing_input_column_raises_error(campaign_config):
    """The recommender should reject incomplete input."""
    df = pd.DataFrame(
        [
            {
                "customer_unique_id": "customer_009",
                "risk_tier": "High Risk",
                "segment_label": "Satisfied One-Time Buyers",
                # value_tier intentionally missing
            }
        ]
    )

    with pytest.raises(KeyError, match="value_tier"):
        recommend_campaigns(df, config=campaign_config)
