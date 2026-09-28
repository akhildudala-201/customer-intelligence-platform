
from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy.engine import Engine

from app.api.cohort_trend_api import schemas as schemas
from app.api.cohort_trend_api  import services as svc

router = APIRouter(tags=["Trends & Cohorts"])

COHORT_PATTERN = r"^\d{4}-(0[1-9]|1[0-2])$"          # e.g. 2017-03
Granularity = Literal["daily", "weekly", "monthly"]


def get_engine() -> Engine:
    """Shared DB engine. Imported lazily so tests can swap it via
    app.dependency_overrides without needing a real .env / MySQL."""
    from app.Database.database import engine
    return engine


# ----------------------------------------------------------------------
# Shared query parameters
# ----------------------------------------------------------------------
class TrendParams:
    def __init__(
        self,
        granularity: Granularity = Query("monthly", description="Time bucket for each point"),
        start_date: Optional[date] = Query(None, description="Include periods on/after this date"),
        end_date: Optional[date] = Query(None, description="Include periods on/before this date"),
    ):
        if start_date and end_date and start_date > end_date:
            raise HTTPException(400, "start_date must be on or before end_date")
        self.granularity, self.start_date, self.end_date = granularity, start_date, end_date


class CohortRange:
    def __init__(
        self,
        cohort_from: Optional[str] = Query(None, pattern=COHORT_PATTERN, description="First cohort, e.g. 2017-01"),
        cohort_to: Optional[str] = Query(None, pattern=COHORT_PATTERN, description="Last cohort, e.g. 2017-12"),
    ):
        if cohort_from and cohort_to and cohort_from > cohort_to:
            raise HTTPException(400, "cohort_from must be on or before cohort_to")
        self.cohort_from, self.cohort_to = cohort_from, cohort_to


def _trend_response(engine: Engine, p: TrendParams, table: str, model) -> dict:
    """Load one trend table with exactly the fields `model` exposes."""
    columns = list(model.model_fields)
    rows = svc.to_records(svc.get_trend(engine, table, p.granularity, columns, p.start_date, p.end_date))
    return {"granularity": p.granularity, "count": len(rows), "data": rows}


# ----------------------------------------------------------------------
# Trends
# ----------------------------------------------------------------------
@router.get("/churn/trend", response_model=schemas.ChurnTrendResponse,
            summary="Churn and retention over time")
def churn_trend(p: TrendParams = Depends(), engine: Engine = Depends(get_engine)):
    """Churn rate per period. `is_censored = true` marks periods whose churn
    is not final yet (the 180-day inactivity window hasn't closed)."""
    return _trend_response(engine, p, svc.CHURN_TABLE, schemas.ChurnTrendPoint)


@router.get("/revenue/trend", response_model=schemas.RevenueTrendResponse,
            summary="Revenue over time")
def revenue_trend(p: TrendParams = Depends(), engine: Engine = Depends(get_engine)):
    return _trend_response(engine, p, svc.REVENUE_TABLE, schemas.RevenueTrendPoint)


@router.get("/trends/combined", response_model=schemas.CombinedTrendResponse,
            summary="Revenue and churn side by side")
def combined_trend(p: TrendParams = Depends(), engine: Engine = Depends(get_engine)):
    return _trend_response(engine, p, svc.COMBINED_TABLE, schemas.CombinedTrendPoint)


@router.get("/trends/new-vs-repeat", response_model=schemas.NewVsRepeatResponse,
            summary="New vs repeat customers over time")
def new_vs_repeat(p: TrendParams = Depends(), engine: Engine = Depends(get_engine)):
    """`repeat_customers` = customers placing a non-first order in the period."""
    rows = svc.to_records(svc.get_new_vs_repeat(engine, p.granularity, p.start_date, p.end_date))
    return {"granularity": p.granularity, "count": len(rows), "data": rows}


# ----------------------------------------------------------------------
# Cohorts  (fixed paths must be declared before /cohort/{cohort})
# ----------------------------------------------------------------------
@router.get("/customers/cohort-retention", response_model=schemas.RetentionMatrixResponse,
            summary="Cohort retention grid (heatmap)")
def cohort_retention(r: CohortRange = Depends(), engine: Engine = Depends(get_engine)):
    df = svc.get_cohorts(engine, r.cohort_from, r.cohort_to)
    return {"generated_date": svc.generated_date(df), **svc.retention_matrix(df)}


def _cohort_list(df, data) -> dict:
    return {"generated_date": svc.generated_date(df), "count": len(data), "data": data}


@router.get("/cohort/churn", response_model=schemas.CohortChurnResponse,
            summary="Churn per cohort")
def cohort_churn(r: CohortRange = Depends(), engine: Engine = Depends(get_engine)):
    df = svc.get_cohorts(engine, r.cohort_from, r.cohort_to)
    data = svc.cohort_series(
        df, ["monthly_churn_rate_pct", "cumulative_churn_rate_pct"],
        lambda rows: {"final_churn_rate_pct": svc.first_value(rows, "final_churn_rate_pct")},
    )
    return _cohort_list(df, data)


@router.get("/cohort/revenue", response_model=schemas.CohortRevenueResponse,
            summary="Cumulative revenue per customer, per cohort")
def cohort_revenue(r: CohortRange = Depends(), engine: Engine = Depends(get_engine)):
    df = svc.get_cohorts(engine, r.cohort_from, r.cohort_to)
    data = svc.cohort_series(
        df, ["cumulative_average_revenue"],
        lambda rows: {"latest_cumulative_average_revenue": svc.last_value(rows, "cumulative_average_revenue")},
    )
    return _cohort_list(df, data)


@router.get("/cohort/repeat-purchase", response_model=schemas.CohortRepeatResponse,
            summary="Repeat purchase rate per cohort")
def cohort_repeat_purchase(r: CohortRange = Depends(), engine: Engine = Depends(get_engine)):
    df = svc.get_cohorts(engine, r.cohort_from, r.cohort_to)
    data = svc.cohort_series(
        df, ["repeat_purchase_rate_pct", "cumulative_repeat_purchase_rate_pct"],
        lambda rows: {"latest_cumulative_repeat_purchase_rate_pct":
                      svc.last_value(rows, "cumulative_repeat_purchase_rate_pct")},
    )
    return _cohort_list(df, data)


@router.get("/cohort/summary", response_model=schemas.CohortSummaryResponse,
            summary="One summary row per cohort")
def cohort_summary(r: CohortRange = Depends(), engine: Engine = Depends(get_engine)):
    df = svc.get_cohorts(engine, r.cohort_from, r.cohort_to)
    return _cohort_list(df, svc.cohort_summary(df))


@router.get("/cohort/{cohort}", response_model=schemas.CohortDetailResponse,
            summary="All metrics for one cohort")
def cohort_detail(
    cohort: str = Path(..., pattern=COHORT_PATTERN, description="Cohort month", examples=["2017-03"]),
    engine: Engine = Depends(get_engine),
):
    detail = svc.cohort_detail(svc.get_cohorts(engine), cohort)
    if detail is None:
        raise HTTPException(404, f"Cohort '{cohort}' not found")
    return detail
