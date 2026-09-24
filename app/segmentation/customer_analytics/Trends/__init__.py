"""Historical revenue and churn trend analysis package."""

try:
    from app.segmentation.customer_analytics.Trends.data_loader import TrendDataLoader
    from app.segmentation.customer_analytics.Trends.export import TrendExporter
    from app.segmentation.customer_analytics.Trends.schemas import (
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
    from app.segmentation.customer_analytics.Trends.trend_engine import HistoricalTrendEngine
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
