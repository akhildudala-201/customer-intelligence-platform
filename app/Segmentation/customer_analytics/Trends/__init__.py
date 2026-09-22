"""
app.Segmentation.customer_analytics.Trends package

Person 5 — Historical Trend Analysis (Schema 6.5, trend half).
Aggregates historical churn trends and historical revenue trends over time (daily/weekly/monthly)
to feed Person 6 (Forecasting).
"""

try:
    from app.Segmentation.customer_analytics.Trends.data_loader import TrendDataLoader
    from app.Segmentation.customer_analytics.Trends.export import TrendExporter
    from app.Segmentation.customer_analytics.Trends.schemas import (
        CHURN_TREND_COLUMNS,
        COMBINED_TREND_COLUMNS,
        REVENUE_TREND_COLUMNS,
        ChurnTrendRecord,
        RevenueTrendRecord,
        Schema65ValidationResult,
        validate_churn_trend_schema,
        validate_combined_trend_schema,
        validate_revenue_trend_schema,
    )
    from app.Segmentation.customer_analytics.Trends.trend_engine import HistoricalTrendEngine
except (ImportError, ModuleNotFoundError):
    from .data_loader import TrendDataLoader
    from .export import TrendExporter
    from .schemas import (
        CHURN_TREND_COLUMNS,
        COMBINED_TREND_COLUMNS,
        REVENUE_TREND_COLUMNS,
        ChurnTrendRecord,
        RevenueTrendRecord,
        Schema65ValidationResult,
        validate_churn_trend_schema,
        validate_combined_trend_schema,
        validate_revenue_trend_schema,
    )
    from .trend_engine import HistoricalTrendEngine

__all__ = [
    "HistoricalTrendEngine",
    "TrendDataLoader",
    "TrendExporter",
    "REVENUE_TREND_COLUMNS",
    "CHURN_TREND_COLUMNS",
    "COMBINED_TREND_COLUMNS",
    "RevenueTrendRecord",
    "ChurnTrendRecord",
    "Schema65ValidationResult",
    "validate_revenue_trend_schema",
    "validate_churn_trend_schema",
    "validate_combined_trend_schema",
]
