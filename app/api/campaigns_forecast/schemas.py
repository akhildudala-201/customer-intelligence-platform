from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, model_validator


# ----------------------------------------------------------------------
# Campaigns
# ----------------------------------------------------------------------
class ActiveCampaign(BaseModel):
    campaign_name: str
    campaign_priority: int
    customer_count: int


class ActiveCampaignsResponse(BaseModel):
    active_campaigns: int
    customers_targeted: int
    high_priority_customers: int  # campaign_priority <= 2
    campaigns: list[ActiveCampaign]


class CampaignSummaryRow(BaseModel):
    campaign_name: str
    campaign_priority: int
    reason_code: str
    customer_count: int


class RecommendationItem(BaseModel):
    customer_unique_id: str
    risk_tier: str
    segment_label: str
    value_tier: str
    campaign_name: str
    campaign_priority: int
    reason_code: str


class RecommendationsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    summary: list[CampaignSummaryRow]  # aggregate counts per campaign (respects the filters)
    items: list[RecommendationItem]


class CampaignCustomer(RecommendationItem):
    churn_probability: float | None = None


class CampaignCustomersResponse(BaseModel):
    campaign_name: str
    total: int
    page: int
    page_size: int
    items: list[CampaignCustomer]


class ReasonCode(BaseModel):
    reason_code: str
    explanation: str


class CampaignRuleOut(BaseModel):
    risk_tier: str
    segment_label: str
    value_tier: str
    campaign_name: str
    campaign_priority: int
    reason_code: str
    explanation: str


class DefaultRuleOut(BaseModel):
    campaign_name: str
    campaign_priority: int
    reason_code: str
    explanation: str


class RulesResponse(BaseModel):
    risk_tiers: list[str]
    segments: list[str]
    value_tiers: list[str]
    total_rules: int
    default_rule: DefaultRuleOut
    rules: list[CampaignRuleOut]


class SegmentCampaign(BaseModel):
    campaign_name: str
    campaign_priority: int
    reason_code: str
    customer_count: int


class SegmentCampaignsRow(BaseModel):
    segment_label: str
    total_customers: int
    campaigns: list[SegmentCampaign]


class EvaluateRequest(BaseModel):
    """Either a customer_unique_id, or all of risk_tier + segment_label + value_tier."""

    customer_unique_id: str | None = None
    risk_tier: str | None = None
    segment_label: str | None = None
    value_tier: str | None = None

    @model_validator(mode="after")
    def _require_id_or_profile(self) -> "EvaluateRequest":
        has_profile = all([self.risk_tier, self.segment_label, self.value_tier])
        if not self.customer_unique_id and not has_profile:
            raise ValueError(
                "Provide customer_unique_id, or all of risk_tier, segment_label and value_tier."
            )
        return self


class EvaluateResponse(BaseModel):
    customer_unique_id: str | None
    risk_tier: str
    segment_label: str
    value_tier: str
    campaign_name: str
    campaign_priority: int
    reason_code: str
    reason: str
    matched_explicit_rule: bool  # False -> fell back to the default rule
    source: str  # "database" (stored recommendation) or "rules" (computed from the YAML rulebook)


# ----------------------------------------------------------------------
# Churn correlations
# ----------------------------------------------------------------------
class BucketRow(BaseModel):
    bucket: str
    customer_count: int
    avg_churn_probability: float | None
    predicted_churn_rate_pct: float | None


class CorrelationResponse(BaseModel):
    dimension: str
    threshold: float  # churn_probability >= threshold counts as "predicted to churn"
    overall_customers: int
    overall_predicted_churn_rate_pct: float | None
    buckets: list[BucketRow]


# ----------------------------------------------------------------------
# Forecasting
# ----------------------------------------------------------------------
class ForecastPoint(BaseModel):
    period_date: datetime
    forecast: float | None
    ci_lower: float | None
    ci_upper: float | None


class ForecastResponse(BaseModel):
    metric: str
    granularity: str | None
    model_used: str | None
    generated_at: datetime | None
    holdout_mape_pct: float | None
    confidence_level: float = 0.95
    horizon: int
    available_periods: int
    data: list[ForecastPoint]


class AccuracyEntry(BaseModel):
    granularity: str | None
    model_used: str | None
    holdout_mape_pct: float | None
    forecast_periods: int
    first_forecast_period: datetime | None
    last_forecast_period: datetime | None
    generated_at: datetime | None


class ForecastAccuracyResponse(BaseModel):
    revenue: AccuracyEntry | None
    churn: AccuracyEntry | None
