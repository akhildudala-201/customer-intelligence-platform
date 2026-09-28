from typing import Any

from pydantic import BaseModel, ConfigDict


class CustomerSchema(BaseModel):
    """
    Customer information returned by customer-level APIs.
    """

    model_config = ConfigDict(extra="allow")

    customer_unique_id: str

    churn_probability: float | None = None

    segment_id: int | None = None

    segment_label: str | None = None

    risk_tier: str | None = None

    cluster_probability: float | None = None

    clv: float | None = None

    value_tier: str | None = None


class CustomersBySegmentResponse(BaseModel):
    total: int
    limit: int
    offset: int

    customers: list[CustomerSchema]



class SegmentProfile(BaseModel):
    segment_id: int | None = None
    segment_label: str | None = None

    customer_count: int

    average_churn_probability: float | None = None

    average_clv: float | None = None


class SegmentProfileResponse(BaseModel):
    segments: list[SegmentProfile]


class SegmentCustomersResponse(BaseModel):
    segment_id: int

    total: int

    limit: int

    offset: int

    customers: list[CustomerSchema]



class SegmentRiskMix(BaseModel):
    segment_id: int | None = None

    segment_label: str | None = None

    risk_tier: str | None = None

    customer_count: int


class SegmentRiskMixResponse(BaseModel):
    risk_mix: list[SegmentRiskMix]



class RiskDistribution(BaseModel):
    risk_tier: str

    customer_count: int

    percentage: float


class RiskSummaryResponse(BaseModel):
    total_customers: int

    average_churn_probability: float

    risk_distribution: list[RiskDistribution]



class AtRiskResponse(BaseModel):
    total: int

    limit: int

    offset: int

    customers: list[CustomerSchema]



class CustomerProfileResponse(BaseModel):
    customer_id: str

    profile: dict[str, Any]


class SegmentSummary(BaseModel):
    segment_id: int | None = None

    segment_label: str | None = None

    customer_count: int

    average_churn_probability: float | None = None

    customer_percentage: float


class SegmentSummaryResponse(BaseModel):
    total_customers: int

    segment_count: int

    segments: list[SegmentSummary]



class SegmentValueMix(BaseModel):
    segment_id: int | None = None

    segment_label: str | None = None

    value_tier: str | None = None

    customer_count: int


class SegmentValueMixResponse(BaseModel):
    value_mix: list[SegmentValueMix]



class RiskThreshold(BaseModel):
    name: str

    upper_bound: float


class RiskThresholdsResponse(BaseModel):
    thresholds: list[RiskThreshold]

    source: str


class CustomerSearchResponse(BaseModel):
    query: str

    total: int

    limit: int

    offset: int

    customers: list[CustomerSchema]
