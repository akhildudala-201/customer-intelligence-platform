
from datetime import date
from typing import Optional

from pydantic import BaseModel


# ----------------------------------------------------------------------
# Trends
# ----------------------------------------------------------------------
class ChurnTrendPoint(BaseModel):
    period_date: date
    active_customers: Optional[int] = None
    new_customers: Optional[int] = None
    repeat_customers: Optional[int] = None
    churned_customers: Optional[int] = None
    churn_rate: Optional[float] = None          # fraction 0-1
    retention_rate: Optional[float] = None      # fraction 0-1
    rolling_churn_rate: Optional[float] = None
    cumulative_churned: Optional[int] = None
    is_censored: Optional[bool] = None          # True = churn not final yet (180-day window still open)


class RevenueTrendPoint(BaseModel):
    period_date: date
    total_revenue: Optional[float] = None
    order_count: Optional[int] = None
    unique_customers: Optional[int] = None
    avg_order_value: Optional[float] = None
    avg_revenue_per_user: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    rolling_revenue_sma_short: Optional[float] = None
    rolling_revenue_sma_long: Optional[float] = None
    cumulative_revenue: Optional[float] = None


class CombinedTrendPoint(BaseModel):
    period_date: date
    total_revenue: Optional[float] = None
    order_count: Optional[int] = None
    unique_customers: Optional[int] = None
    avg_order_value: Optional[float] = None
    avg_revenue_per_user: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    active_customers: Optional[int] = None
    new_customers: Optional[int] = None
    repeat_customers: Optional[int] = None
    churned_customers: Optional[int] = None
    churn_rate: Optional[float] = None
    retention_rate: Optional[float] = None
    rolling_churn_rate: Optional[float] = None
    is_censored: Optional[bool] = None


class NewVsRepeatPoint(BaseModel):
    period_date: date
    new_customers: Optional[int] = None
    repeat_customers: Optional[int] = None
    repeat_share_pct: Optional[float] = None    # repeat / (new + repeat) * 100


class TrendResponse(BaseModel):
    granularity: str
    count: int
    data: list


class ChurnTrendResponse(TrendResponse):
    data: list[ChurnTrendPoint]


class RevenueTrendResponse(TrendResponse):
    data: list[RevenueTrendPoint]


class CombinedTrendResponse(TrendResponse):
    data: list[CombinedTrendPoint]


class NewVsRepeatResponse(TrendResponse):
    data: list[NewVsRepeatPoint]


# ----------------------------------------------------------------------
# Cohorts  (cohort = "2017-03", label = "Mar 2017", month = "M0", "M1", ...)
# ----------------------------------------------------------------------
class RetentionMatrixResponse(BaseModel):
    """Heatmap-ready: values[i][j] = retention % of cohorts[i] in months[j]."""
    generated_date: Optional[date] = None
    cohorts: list[str]
    labels: list[str]
    months: list[str]
    values: list[list[Optional[float]]]         # None = cohort too recent for that month


class ChurnMonth(BaseModel):
    month: str
    monthly_churn_rate_pct: Optional[float] = None
    cumulative_churn_rate_pct: Optional[float] = None


class RevenueMonth(BaseModel):
    month: str
    cumulative_average_revenue: Optional[float] = None


class RepeatMonth(BaseModel):
    month: str
    repeat_purchase_rate_pct: Optional[float] = None
    cumulative_repeat_purchase_rate_pct: Optional[float] = None


class CohortChurn(BaseModel):
    cohort: str
    label: str
    final_churn_rate_pct: Optional[float] = None
    months: list[ChurnMonth]


class CohortRevenue(BaseModel):
    cohort: str
    label: str
    latest_cumulative_average_revenue: Optional[float] = None
    months: list[RevenueMonth]


class CohortRepeat(BaseModel):
    cohort: str
    label: str
    latest_cumulative_repeat_purchase_rate_pct: Optional[float] = None
    months: list[RepeatMonth]


class CohortListResponse(BaseModel):
    generated_date: Optional[date] = None
    count: int
    data: list


class CohortChurnResponse(CohortListResponse):
    data: list[CohortChurn]


class CohortRevenueResponse(CohortListResponse):
    data: list[CohortRevenue]


class CohortRepeatResponse(CohortListResponse):
    data: list[CohortRepeat]


class CohortSummaryRow(BaseModel):
    cohort: str
    label: str
    cohort_size: Optional[int] = None           # filled once Person 4 adds the column
    months_observed: int
    m1_retention_pct: Optional[float] = None
    m3_retention_pct: Optional[float] = None
    final_churn_rate_pct: Optional[float] = None
    cumulative_repeat_purchase_rate_pct: Optional[float] = None
    cumulative_average_revenue: Optional[float] = None


class CohortSummaryResponse(CohortListResponse):
    data: list[CohortSummaryRow]


class CohortMonthDetail(BaseModel):
    month: str
    retention_rate_pct: Optional[float] = None
    repeat_purchase_rate_pct: Optional[float] = None
    cumulative_repeat_purchase_rate_pct: Optional[float] = None
    monthly_churn_rate_pct: Optional[float] = None
    cumulative_churn_rate_pct: Optional[float] = None
    cumulative_average_revenue: Optional[float] = None


class CohortDetailResponse(BaseModel):
    cohort: str
    label: str
    cohort_size: Optional[int] = None
    final_churn_rate_pct: Optional[float] = None
    generated_date: Optional[date] = None
    months: list[CohortMonthDetail]
