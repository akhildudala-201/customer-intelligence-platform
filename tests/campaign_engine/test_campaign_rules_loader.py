import pytest

from app.segmentation.customer_intelligence.campaign_engine.campaign_rules_loader import (
    CampaignRule,
    CampaignRulesConfig,
    load_campaign_rules_config,
)


def test_campaign_rules_config_loads():
    """The real campaign YAML should load successfully."""
    load_campaign_rules_config.cache_clear()

    config = load_campaign_rules_config()

    assert isinstance(config, CampaignRulesConfig)


def test_required_config_sections_are_loaded():
    """Verify all required configuration sections are present."""
    load_campaign_rules_config.cache_clear()

    config = load_campaign_rules_config()

    assert config.risk_tiers
    assert config.segments
    assert config.value_tiers
    assert config.reason_codes
    assert config.default_rule
    assert config.rules


def test_risk_tiers_are_loaded_correctly():
    """Verify the configured risk tiers."""
    load_campaign_rules_config.cache_clear()

    config = load_campaign_rules_config()

    assert config.risk_tiers == (
        "High Risk",
        "Medium Risk",
        "Low Risk",
    )


def test_value_tiers_are_loaded_correctly():
    """Verify the configured value tiers."""
    load_campaign_rules_config.cache_clear()

    config = load_campaign_rules_config()

    assert config.value_tiers == (
        "High",
        "Medium",
        "Low",
    )


def test_segments_are_loaded_correctly():
    """Verify the three segments currently declared in the YAML."""
    load_campaign_rules_config.cache_clear()

    config = load_campaign_rules_config()

    assert "High-Value Satisfied Repeat Buyers" in config.segments
    assert "Satisfied One-Time Buyers" in config.segments
    assert "Product-Dissatisfied One-Time Buyers" in config.segments


def test_reason_codes_are_loaded():
    """Verify all configured reason codes exist."""
    load_campaign_rules_config.cache_clear()

    config = load_campaign_rules_config()

    expected_codes = {"RC00", "RC01", "RC02", "RC03", "RC04"}

    assert expected_codes.issubset(set(config.reason_codes))


def test_default_rule_is_loaded_correctly():
    """Verify the fallback campaign configuration."""
    load_campaign_rules_config.cache_clear()

    config = load_campaign_rules_config()

    assert isinstance(config.default_rule, CampaignRule)
    assert config.default_rule.campaign_name == (
        "Standard Customer Engagement Program"
    )
    assert config.default_rule.campaign_priority == 5
    assert config.default_rule.reason_code == "RC00"


def test_rules_have_valid_campaign_priorities():
    """Every campaign priority must be between 1 and 5."""
    load_campaign_rules_config.cache_clear()

    config = load_campaign_rules_config()

    for rule in config.rules.values():
        assert 1 <= rule.campaign_priority <= 5


def test_rules_have_valid_reason_codes():
    """Every rule must use a declared reason code."""
    load_campaign_rules_config.cache_clear()

    config = load_campaign_rules_config()

    for rule in config.rules.values():
        assert rule.reason_code in config.reason_codes


def test_explicit_rules_use_declared_values():
    """Every rule key must use declared risk, segment and value tiers."""
    load_campaign_rules_config.cache_clear()

    config = load_campaign_rules_config()

    for risk_tier, segment, value_tier in config.rules:
        assert risk_tier in config.risk_tiers
        assert segment in config.segments
        assert value_tier in config.value_tiers


def test_duplicate_rule_detection(tmp_path):
    """
    The loader should reject duplicate
    (risk_tier, segment, value_tier) combinations.
    """
    config_file = tmp_path / "duplicate_rules.yaml"

    config_file.write_text(
        """
risk_tiers:
  - High Risk

segments:
  - Test Segment

value_tiers:
  - High

reason_codes:
  RC00: "General"

default_rule:
  campaign_name: "Default Campaign"
  campaign_priority: 5
  reason_code: RC00

rules:
  - risk_tier: High Risk
    segment: Test Segment
    value_tier: High
    campaign_name: "Campaign A"
    campaign_priority: 1
    reason_code: RC00

  - risk_tier: High Risk
    segment: Test Segment
    value_tier: High
    campaign_name: "Campaign B"
    campaign_priority: 2
    reason_code: RC00
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicates an earlier rule"):
        load_campaign_rules_config(config_file)


def test_invalid_campaign_priority_is_rejected(tmp_path):
    """The loader should reject priorities outside 1-5."""
    config_file = tmp_path / "invalid_priority.yaml"

    config_file.write_text(
        """
risk_tiers:
  - High Risk

segments:
  - Test Segment

value_tiers:
  - High

reason_codes:
  RC00: "General"

default_rule:
  campaign_name: "Default Campaign"
  campaign_priority: 5
  reason_code: RC00

rules:
  - risk_tier: High Risk
    segment: Test Segment
    value_tier: High
    campaign_name: "Test Campaign"
    campaign_priority: 6
    reason_code: RC00
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 1-5"):
        load_campaign_rules_config(config_file)
