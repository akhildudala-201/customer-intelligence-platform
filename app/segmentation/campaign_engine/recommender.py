

from __future__ import annotations

import pandas as pd

from app.segmentation.customer_intelligence.campaign_engine.campaign_rules_loader import (
    CampaignRulesConfig,
    load_campaign_rules_config,
)

ID_COLUMN = "customer_unique_id"
RISK_TIER_COLUMN = "risk_tier"
SEGMENT_LABEL_COLUMN = "segment_label"
VALUE_TIER_COLUMN = "value_tier"

CAMPAIGN_NAME_COLUMN = "campaign_name"
CAMPAIGN_PRIORITY_COLUMN = "campaign_priority"
REASON_CODE_COLUMN = "reason_code"

_INPUT_COLUMNS = [ID_COLUMN, RISK_TIER_COLUMN, SEGMENT_LABEL_COLUMN, VALUE_TIER_COLUMN]
OUTPUT_COLUMNS = _INPUT_COLUMNS + [CAMPAIGN_NAME_COLUMN, CAMPAIGN_PRIORITY_COLUMN, REASON_CODE_COLUMN]


def recommend_campaigns(
    scoreable_df: pd.DataFrame, config: CampaignRulesConfig | None = None
) -> pd.DataFrame:
    """
    Schema 6.3 output: customer_unique_id, risk_tier, segment_label,
    value_tier, campaign_name, campaign_priority, reason_code.

    `scoreable_df` needs at least [customer_unique_id, risk_tier,
    segment_label, value_tier] — campaign_input_repository.build_campaign_input_base()
    produces exactly that (plus clv, which passes through unused here).
    Pass `config` explicitly in tests to avoid touching the YAML file on
    disk; otherwise the cached campaign_rules_reasoncodes.yaml config is
    loaded automatically.
    """
    missing = [c for c in _INPUT_COLUMNS if c not in scoreable_df.columns]
    if missing:
        raise KeyError(f"recommend_campaigns() input is missing column(s) {missing}.")

    config = config or load_campaign_rules_config()
    _warn_on_unexpected_values(scoreable_df, config)

    campaign_names: list[str] = []
    campaign_priorities: list[int] = []
    reason_codes: list[str] = []
    matched_explicit = 0

    for row in scoreable_df.itertuples(index=False):
        rule, was_explicit = config.lookup(
            getattr(row, RISK_TIER_COLUMN),
            getattr(row, SEGMENT_LABEL_COLUMN),
            getattr(row, VALUE_TIER_COLUMN),
        )
        campaign_names.append(rule.campaign_name)
        campaign_priorities.append(rule.campaign_priority)
        reason_codes.append(rule.reason_code)
        matched_explicit += was_explicit

    result = scoreable_df[_INPUT_COLUMNS].copy()
    result[CAMPAIGN_NAME_COLUMN] = campaign_names
    result[CAMPAIGN_PRIORITY_COLUMN] = campaign_priorities
    result[REASON_CODE_COLUMN] = reason_codes

    _print_match_summary(result, matched_explicit)
    return result


def _warn_on_unexpected_values(scoreable_df: pd.DataFrame, config: CampaignRulesConfig) -> None:
    """Printed, not fatal — see module docstring. A value outside the
    config's declared lists still gets a result (default_rule, via
    CampaignRulesConfig.lookup's fallback), it just means the config
    file is stale relative to the current pipeline output and should be
    updated (see campaign_rules_reasoncodes.yaml's own "IF K CHANGES"
    note)."""
    # (dataframe column, config values, the YAML top-level key that declares them)
    checks = (
        (RISK_TIER_COLUMN, config.risk_tiers, "risk_tiers"),
        (SEGMENT_LABEL_COLUMN, config.segments, "segments"),
        (VALUE_TIER_COLUMN, config.value_tiers, "value_tiers"),
    )
    for column, allowed, yaml_key in checks:
        unexpected = sorted(set(scoreable_df[column].unique()) - set(allowed))
        if unexpected:
            print(
                f"  WARNING: {len(unexpected)} `{column}` value(s) not declared in "
                f"campaign_rules_reasoncodes.yaml: {unexpected}. These customers "
                "will fall back to default_rule. Update the YAML's "
                f"`{yaml_key}:` list and add explicit rules if this is expected "
                "(e.g. GMM was rerun with a different k)."
            )


def _print_match_summary(result: pd.DataFrame, matched_explicit: int) -> None:
    """Printed for review, same convention as gmm's Step 5/6 profile
    output — lets a human see the fallback rate and the
    campaign/reason_code distribution in one run rather than trusting
    the lookup blindly."""
    total = len(result)
    fell_back = total - matched_explicit
    print(
        f"  -> {total} customer(s) scored: {matched_explicit} matched an "
        f"explicit rule, {fell_back} fell back to default_rule "
        f"({fell_back / total:.1%})." if total else "  -> 0 customers scored."
    )
    print("  Campaign distribution:")
    print(result[CAMPAIGN_NAME_COLUMN].value_counts().to_string())
    print("  Reason code distribution:")
    print(result[REASON_CODE_COLUMN].value_counts().to_string())
