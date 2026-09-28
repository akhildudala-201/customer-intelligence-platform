from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import text

from app.Database.database import engine

RECOMMENDATIONS_TABLE = "customer_campaign_recommendations"
PREDICTIONS_TABLE = "churn_predictions"
FEATURES_TABLE = "customer_features"
FORECAST_REVENUE_TABLE = "forecast_revenue_trend"
FORECAST_CHURN_TABLE = "forecast_churn_trend"

_FILTER_COLUMNS = {
    "campaign_name",
    "campaign_priority",
    "risk_tier",
    "segment_label",
    "value_tier",
}


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _clean(value: Any) -> Any:
    return float(value) if isinstance(value, Decimal) else value


def fetch_all(sql: str, params: dict | None = None) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params or {}).mappings().all()
    return [{key: _clean(val) for key, val in row.items()} for row in rows]


def fetch_one(sql: str, params: dict | None = None) -> dict | None:
    rows = fetch_all(sql, params)
    return rows[0] if rows else None


def _where(filters: dict[str, Any]) -> tuple[str, dict]:
    clauses, params = [], {}
    for column, value in filters.items():
        if value is None:
            continue
        if column not in _FILTER_COLUMNS:
            raise ValueError(f"Unsupported filter column: {column}")
        clauses.append(f"r.{column} = :{column}")
        params[column] = value
    return ("WHERE " + " AND ".join(clauses)) if clauses else "", params


# ----------------------------------------------------------------------
# campaigns
# ----------------------------------------------------------------------
def active_campaigns() -> list[dict]:
    return fetch_all(
        f"""
        SELECT campaign_name,
               MIN(campaign_priority) AS campaign_priority,
               COUNT(*)               AS customer_count
        FROM {RECOMMENDATIONS_TABLE}
        GROUP BY campaign_name
        ORDER BY campaign_priority, customer_count DESC
        """
    )


def recommendation_summary(filters: dict[str, Any]) -> list[dict]:
    where, params = _where(filters)
    return fetch_all(
        f"""
        SELECT r.campaign_name, r.campaign_priority, r.reason_code,
               COUNT(*) AS customer_count
        FROM {RECOMMENDATIONS_TABLE} r
        {where}
        GROUP BY r.campaign_name, r.campaign_priority, r.reason_code
        ORDER BY r.campaign_priority, customer_count DESC
        """,
        params,
    )


def recommendation_count(filters: dict[str, Any]) -> int:
    where, params = _where(filters)
    row = fetch_one(f"SELECT COUNT(*) AS n FROM {RECOMMENDATIONS_TABLE} r {where}", params)
    return int(row["n"]) if row else 0


def recommendation_page(filters: dict[str, Any], limit: int, offset: int) -> list[dict]:
    where, params = _where(filters)
    return fetch_all(
        f"""
        SELECT r.customer_unique_id, r.risk_tier, r.segment_label, r.value_tier,
               r.campaign_name, r.campaign_priority, r.reason_code
        FROM {RECOMMENDATIONS_TABLE} r
        {where}
        ORDER BY r.campaign_priority, r.customer_unique_id
        LIMIT :limit OFFSET :offset
        """,
        {**params, "limit": limit, "offset": offset},
    )


def campaign_customers_page(campaign_name: str, limit: int, offset: int) -> list[dict]:
    """Customers for one campaign, highest churn probability first."""
    return fetch_all(
        f"""
        SELECT r.customer_unique_id, r.risk_tier, r.segment_label, r.value_tier,
               r.campaign_name, r.campaign_priority, r.reason_code,
               p.churn_probability
        FROM {RECOMMENDATIONS_TABLE} r
        LEFT JOIN {PREDICTIONS_TABLE} p
               ON p.customer_unique_id = r.customer_unique_id
        WHERE r.campaign_name = :campaign_name
        ORDER BY p.churn_probability DESC, r.customer_unique_id
        LIMIT :limit OFFSET :offset
        """,
        {"campaign_name": campaign_name, "limit": limit, "offset": offset},
    )


def campaign_customer_count(campaign_name: str) -> int:
    row = fetch_one(
        f"SELECT COUNT(*) AS n FROM {RECOMMENDATIONS_TABLE} WHERE campaign_name = :n",
        {"n": campaign_name},
    )
    return int(row["n"]) if row else 0


def campaigns_by_segment() -> list[dict]:
    return fetch_all(
        f"""
        SELECT segment_label, campaign_name, campaign_priority, reason_code,
               COUNT(*) AS customer_count
        FROM {RECOMMENDATIONS_TABLE}
        GROUP BY segment_label, campaign_name, campaign_priority, reason_code
        ORDER BY segment_label, campaign_priority, customer_count DESC
        """
    )


def recommendation_for_customer(customer_unique_id: str) -> dict | None:
    return fetch_one(
        f"""
        SELECT customer_unique_id, risk_tier, segment_label, value_tier,
               campaign_name, campaign_priority, reason_code
        FROM {RECOMMENDATIONS_TABLE}
        WHERE customer_unique_id = :cid
        LIMIT 1
        """,
        {"cid": customer_unique_id},
    )


# ----------------------------------------------------------------------
# churn correlations
# ----------------------------------------------------------------------
# avg_delivery_delay_days = delivered - estimated date (negative = early).
# NOTE: build_features zero-fills missing delay values, so customers without
# delivery data land in the "on time / early" bucket.
_DELIVERY_BUCKET = """
    CASE
        WHEN f.avg_delivery_delay_days <= -7 THEN '7+ days early'
        WHEN f.avg_delivery_delay_days <= 0  THEN 'On time / up to 7 days early'
        WHEN f.avg_delivery_delay_days <= 3  THEN '1-3 days late'
        WHEN f.avg_delivery_delay_days <= 7  THEN '4-7 days late'
        ELSE '8+ days late'
    END
"""
_DELIVERY_ORDER = """
    CASE
        WHEN f.avg_delivery_delay_days <= -7 THEN 1
        WHEN f.avg_delivery_delay_days <= 0  THEN 2
        WHEN f.avg_delivery_delay_days <= 3  THEN 3
        WHEN f.avg_delivery_delay_days <= 7  THEN 4
        ELSE 5
    END
"""
_REVIEW_BUCKET = """
    CASE
        WHEN f.avg_review_score IS NULL THEN 'No review'
        WHEN f.avg_review_score < 2     THEN 'Poor (<2)'
        WHEN f.avg_review_score < 3     THEN 'Below average (2-3)'
        WHEN f.avg_review_score < 4     THEN 'Average (3-4)'
        WHEN f.avg_review_score < 5     THEN 'Good (4-5)'
        ELSE 'Perfect (5)'
    END
"""
_REVIEW_ORDER = """
    CASE
        WHEN f.avg_review_score IS NULL THEN 6
        WHEN f.avg_review_score < 2     THEN 1
        WHEN f.avg_review_score < 3     THEN 2
        WHEN f.avg_review_score < 4     THEN 3
        WHEN f.avg_review_score < 5     THEN 4
        ELSE 5
    END
"""


def _churn_by_bucket(bucket_sql: str, order_sql: str, threshold: float) -> list[dict]:
    return fetch_all(
        f"""
        SELECT bucket,
               COUNT(*)                                        AS customer_count,
               AVG(churn_probability)                          AS avg_churn_probability,
               100.0 * AVG(churn_probability >= :threshold)    AS predicted_churn_rate_pct
        FROM (
            SELECT p.churn_probability,
                   {bucket_sql} AS bucket,
                   {order_sql}  AS bucket_order
            FROM {FEATURES_TABLE} f
            JOIN {PREDICTIONS_TABLE} p
              ON p.customer_unique_id = f.customer_unique_id
        ) t
        GROUP BY bucket, bucket_order
        ORDER BY bucket_order
        """,
        {"threshold": threshold},
    )


def churn_by_delivery_delay(threshold: float) -> list[dict]:
    return _churn_by_bucket(_DELIVERY_BUCKET, _DELIVERY_ORDER, threshold)


def churn_by_review_score(threshold: float) -> list[dict]:
    return _churn_by_bucket(_REVIEW_BUCKET, _REVIEW_ORDER, threshold)


# ----------------------------------------------------------------------
# forecasts
# ----------------------------------------------------------------------
def forecast_revenue() -> list[dict]:
    return fetch_all(
        f"""
        SELECT period_date, granularity, generated_at, model_used, holdout_mape_pct,
               forecasted_revenue AS forecast,
               ci_lower_revenue   AS ci_lower,
               ci_upper_revenue   AS ci_upper
        FROM {FORECAST_REVENUE_TABLE}
        ORDER BY period_date
        """
    )


def forecast_churn() -> list[dict]:
    return fetch_all(
        f"""
        SELECT period_date, granularity, generated_at, model_used, holdout_mape_pct,
               forecasted_churn_rate AS forecast,
               ci_lower_churn        AS ci_lower,
               ci_upper_churn        AS ci_upper
        FROM {FORECAST_CHURN_TABLE}
        ORDER BY period_date
        """
    )
