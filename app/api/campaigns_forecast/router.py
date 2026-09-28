from __future__ import annotations

import logging
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.exc import OperationalError, ProgrammingError

from app.api.campaigns_forecast import repository as repo
from app.api.campaigns_forecast.schemas import (
    AccuracyEntry,
    ActiveCampaign,
    ActiveCampaignsResponse,
    BucketRow,
    CampaignCustomer,
    CampaignCustomersResponse,
    CampaignRuleOut,
    CampaignSummaryRow,
    CorrelationResponse,
    DefaultRuleOut,
    EvaluateRequest,
    EvaluateResponse,
    ForecastAccuracyResponse,
    ForecastPoint,
    ForecastResponse,
    ReasonCode,
    RecommendationItem,
    RecommendationsResponse,
    RulesResponse,
    SegmentCampaign,
    SegmentCampaignsRow,
)
from app.segmentation.customer_intelligence.campaign_engine.campaign_rules_loader import (
    load_campaign_rules_config,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Campaigns, churn correlations & forecasting"])

HIGH_PRIORITY_MAX = 2  # campaign_priority 1-2 counts as "high priority"


def _db(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:

    try:
        return fn(*args, **kwargs)
    except (ProgrammingError, OperationalError) as exc:
        logger.exception("Database error in %s", getattr(fn, "__name__", fn))
        raise HTTPException(
            status_code=503,
            detail=(
                "A required table is missing or the database is unavailable. "
                "Run scripts/run_pipeline.py so the pipeline tables exist. "
                f"({exc.orig})"
            ),
        ) from exc


def _explain(code: str) -> str:
    return load_campaign_rules_config().reason_codes.get(code, "")


# ======================================================================
# Campaigns
# ======================================================================
@router.get("/campaigns/active", response_model=ActiveCampaignsResponse)
def campaigns_active():
 
    rows = _db(repo.active_campaigns)
    campaigns = [ActiveCampaign(**row) for row in rows]
    return ActiveCampaignsResponse(
        active_campaigns=len(campaigns),
        customers_targeted=sum(c.customer_count for c in campaigns),
        high_priority_customers=sum(
            c.customer_count for c in campaigns if c.campaign_priority <= HIGH_PRIORITY_MAX
        ),
        campaigns=campaigns,
    )


@router.get("/campaigns/recommendations", response_model=RecommendationsResponse)
def campaigns_recommendations(
    campaign_name: str | None = None,
    risk_tier: str | None = None,
    segment_label: str | None = None,
    value_tier: str | None = None,
    campaign_priority: int | None = Query(None, ge=1, le=5),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    """Per-customer campaign / priority / reason_code, plus aggregate counts per campaign."""
    filters = {
        "campaign_name": campaign_name,
        "risk_tier": risk_tier,
        "segment_label": segment_label,
        "value_tier": value_tier,
        "campaign_priority": campaign_priority,
    }
    total = _db(repo.recommendation_count, filters)
    summary = _db(repo.recommendation_summary, filters)
    items = _db(repo.recommendation_page, filters, page_size, (page - 1) * page_size)
    return RecommendationsResponse(
        total=total,
        page=page,
        page_size=page_size,
        summary=[CampaignSummaryRow(**row) for row in summary],
        items=[RecommendationItem(**row) for row in items],
    )


@router.get("/campaigns/reason-codes", response_model=list[ReasonCode])
def campaigns_reason_codes():
    """reason_code -> human-readable explanation (from campaign_rules_reasoncodes.yaml)."""
    config = load_campaign_rules_config()
    return [ReasonCode(reason_code=code, explanation=text) for code, text in config.reason_codes.items()]


@router.get("/campaigns/rules", response_model=RulesResponse)
def campaigns_rules():
    """The full rulebook: (risk tier, segment, value tier) -> campaign, priority, reason."""
    config = load_campaign_rules_config()
    risk_order = {name: i for i, name in enumerate(config.risk_tiers)}
    rules = [
        CampaignRuleOut(
            risk_tier=risk,
            segment_label=segment,
            value_tier=value,
            campaign_name=rule.campaign_name,
            campaign_priority=rule.campaign_priority,
            reason_code=rule.reason_code,
            explanation=_explain(rule.reason_code),
        )
        for (risk, segment, value), rule in config.rules.items()
    ]
    rules.sort(key=lambda r: (r.campaign_priority, r.segment_label, risk_order.get(r.risk_tier, 99)))
    default = config.default_rule
    return RulesResponse(
        risk_tiers=list(config.risk_tiers),
        segments=list(config.segments),
        value_tiers=list(config.value_tiers),
        total_rules=len(rules),
        default_rule=DefaultRuleOut(
            campaign_name=default.campaign_name,
            campaign_priority=default.campaign_priority,
            reason_code=default.reason_code,
            explanation=_explain(default.reason_code),
        ),
        rules=rules,
    )


@router.get("/campaigns/by-segment", response_model=list[SegmentCampaignsRow])
def campaigns_by_segment():
    """Which campaigns each customer segment is getting, with counts (for a stacked bar / table)."""
    rows = _db(repo.campaigns_by_segment)
    grouped: dict[str, SegmentCampaignsRow] = {}
    for row in rows:
        entry = grouped.setdefault(
            row["segment_label"],
            SegmentCampaignsRow(segment_label=row["segment_label"], total_customers=0, campaigns=[]),
        )
        entry.total_customers += row["customer_count"]
        entry.campaigns.append(
            SegmentCampaign(
                campaign_name=row["campaign_name"],
                campaign_priority=row["campaign_priority"],
                reason_code=row["reason_code"],
                customer_count=row["customer_count"],
            )
        )
    return sorted(grouped.values(), key=lambda s: s.total_customers, reverse=True)


@router.get("/campaigns/{campaign_name}/customers", response_model=CampaignCustomersResponse)
def campaign_customers(
    campaign_name: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    """Customers targeted by one campaign (highest churn probability first) for drill-down."""
    total = _db(repo.campaign_customer_count, campaign_name)
    if total == 0:
        raise HTTPException(status_code=404, detail=f"No customers found for campaign: {campaign_name}")
    rows = _db(repo.campaign_customers_page, campaign_name, page_size, (page - 1) * page_size)
    return CampaignCustomersResponse(
        campaign_name=campaign_name,
        total=total,
        page=page,
        page_size=page_size,
        items=[CampaignCustomer(**row) for row in rows],
    )


@router.post("/campaigns/evaluate", response_model=EvaluateResponse)
def campaigns_evaluate(payload: EvaluateRequest):
    """Suggest a campaign (and why) for one customer.

    - With `customer_unique_id`: returns the stored recommendation; if the customer isn't
      in the table but a full profile was also sent, falls back to the rulebook.
    - With `risk_tier` + `segment_label` + `value_tier`: applies the YAML rulebook directly
      (useful for "what if" checks from the dashboard).
    """
    config = load_campaign_rules_config()
    has_profile = all([payload.risk_tier, payload.segment_label, payload.value_tier])

    stored = _db(repo.recommendation_for_customer, payload.customer_unique_id) if payload.customer_unique_id else None

    if stored:
        _, explicit = config.lookup(stored["risk_tier"], stored["segment_label"], stored["value_tier"])
        return EvaluateResponse(
            **stored,
            reason=_explain(stored["reason_code"]),
            matched_explicit_rule=explicit,
            source="database",
        )

    if not has_profile:
        raise HTTPException(
            status_code=404,
            detail=f"No campaign recommendation found for customer_unique_id: {payload.customer_unique_id}",
        )

    rule, explicit = config.lookup(payload.risk_tier, payload.segment_label, payload.value_tier)
    return EvaluateResponse(
        customer_unique_id=payload.customer_unique_id,
        risk_tier=payload.risk_tier,
        segment_label=payload.segment_label,
        value_tier=payload.value_tier,
        campaign_name=rule.campaign_name,
        campaign_priority=rule.campaign_priority,
        reason_code=rule.reason_code,
        reason=_explain(rule.reason_code),
        matched_explicit_rule=explicit,
        source="rules",
    )


# ======================================================================
# Churn correlations
# ======================================================================
def _correlation_response(dimension: str, rows: list[dict], threshold: float) -> CorrelationResponse:
    buckets = [BucketRow(**row) for row in rows]
    total = sum(b.customer_count for b in buckets)
    weighted = sum((b.predicted_churn_rate_pct or 0) * b.customer_count for b in buckets)
    return CorrelationResponse(
        dimension=dimension,
        threshold=threshold,
        overall_customers=total,
        overall_predicted_churn_rate_pct=round(weighted / total, 2) if total else None,
        buckets=buckets,
    )


@router.get("/churn/delivery-correlation", response_model=CorrelationResponse)
def churn_delivery_correlation(
    threshold: float = Query(0.5, ge=0, le=1, description="churn_probability >= threshold = predicted to churn"),
):
    """Churn by delivery-delay bucket (delivered vs. estimated date)."""
    rows = _db(repo.churn_by_delivery_delay, threshold)
    return _correlation_response("delivery_delay", rows, threshold)


@router.get("/churn/review-correlation", response_model=CorrelationResponse)
def churn_review_correlation(
    threshold: float = Query(0.5, ge=0, le=1, description="churn_probability >= threshold = predicted to churn"),
):
    """Churn by average review-score bucket."""
    rows = _db(repo.churn_by_review_score, threshold)
    return _correlation_response("review_score", rows, threshold)


# ======================================================================
# Forecasting
# ======================================================================
def _forecast_response(metric: str, rows: list[dict], horizon: int) -> ForecastResponse:
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No {metric} forecast found. Run the forecasting step of scripts/run_pipeline.py.",
        )
    head = rows[0]
    return ForecastResponse(
        metric=metric,
        granularity=head.get("granularity"),
        model_used=head.get("model_used"),
        generated_at=head.get("generated_at"),
        holdout_mape_pct=head.get("holdout_mape_pct"),
        horizon=min(horizon, len(rows)),
        available_periods=len(rows),
        data=[
            ForecastPoint(
                period_date=r["period_date"],
                forecast=r["forecast"],
                ci_lower=r["ci_lower"],
                ci_upper=r["ci_upper"],
            )
            for r in rows[:horizon]
        ],
    )


@router.get("/forecast/churn", response_model=ForecastResponse)
def forecast_churn(horizon: int = Query(6, ge=1, le=60, description="Number of future periods")):
    """Forecast churn rate (0-1) for the next N periods with a 95% confidence interval."""
    return _forecast_response("churn_rate", _db(repo.forecast_churn), horizon)


@router.get("/forecast/revenue", response_model=ForecastResponse)
def forecast_revenue(horizon: int = Query(6, ge=1, le=60, description="Number of future periods")):
    """Forecast revenue for the next N periods with a 95% confidence interval."""
    return _forecast_response("revenue", _db(repo.forecast_revenue), horizon)


def _accuracy_entry(rows: list[dict]) -> AccuracyEntry | None:
    if not rows:
        return None
    head = rows[0]
    return AccuracyEntry(
        granularity=head.get("granularity"),
        model_used=head.get("model_used"),
        holdout_mape_pct=head.get("holdout_mape_pct"),
        forecast_periods=len(rows),
        first_forecast_period=rows[0]["period_date"],
        last_forecast_period=rows[-1]["period_date"],
        generated_at=head.get("generated_at"),
    )


@router.get("/forecast/accuracy", response_model=ForecastAccuracyResponse)
def forecast_accuracy():
    """Holdout MAPE and model used for each forecast (lower MAPE = more trustworthy)."""
    return ForecastAccuracyResponse(
        revenue=_accuracy_entry(_db(repo.forecast_revenue)),
        churn=_accuracy_entry(_db(repo.forecast_churn)),
    )
