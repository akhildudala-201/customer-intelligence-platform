"""
app/Trends/schemas.py

Schema 6.5 Specification for Historical Trend Analysis (Person 5 -> Person 6 Contract).
Provides data validation schemas, column specifications, and validation utilities
for Revenue, Churn, and Combined historical time-series datasets.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd


# Expected column schemas for Person 6 Consumption (Schema 6.5)
REVENUE_TREND_COLUMNS: List[str] = [
    "granularity",
    "period_date",
    "total_revenue",
    "order_count",
    "unique_customers",
    "avg_order_value",
    "avg_revenue_per_user",
    "revenue_growth_pct",
    "rolling_revenue_sma_short",
    "rolling_revenue_sma_long",
    "cumulative_revenue",
    "is_forecast",
]

CHURN_TREND_COLUMNS: List[str] = [
    "granularity",
    "period_date",
    "active_customers",
    "new_customers",
    "repeat_customers",
    "churned_customers",
    "churn_rate",
    "retention_rate",
    "rolling_churn_rate",
    "cumulative_churned",
    "is_forecast",
]

COMBINED_TREND_COLUMNS: List[str] = [
    "granularity",
    "period_date",
    "total_revenue",
    "order_count",
    "unique_customers",
    "avg_order_value",
    "avg_revenue_per_user",
    "revenue_growth_pct",
    "active_customers",
    "new_customers",
    "repeat_customers",
    "churned_customers",
    "churn_rate",
    "retention_rate",
    "rolling_churn_rate",
    "is_forecast",
]


@dataclass
class RevenueTrendRecord:
    """Dataclass representing a single row in Schema 6.5 Revenue Trend."""
    granularity: str
    period_date: pd.Timestamp
    total_revenue: float
    order_count: int
    unique_customers: int
    avg_order_value: float
    avg_revenue_per_user: float
    revenue_growth_pct: Optional[float] = None
    rolling_revenue_sma_short: Optional[float] = None
    rolling_revenue_sma_long: Optional[float] = None
    cumulative_revenue: float = 0.0
    is_forecast: bool = False


@dataclass
class ChurnTrendRecord:
    """Dataclass representing a single row in Schema 6.5 Churn Trend."""
    granularity: str
    period_date: pd.Timestamp
    active_customers: int
    new_customers: int
    repeat_customers: int
    churned_customers: int
    churn_rate: float
    retention_rate: float
    rolling_churn_rate: Optional[float] = None
    cumulative_churned: int = 0
    is_forecast: bool = False


@dataclass
class Schema65ValidationResult:
    """Validation report for Schema 6.5 compliance."""
    is_valid: bool
    missing_columns: List[str] = field(default_factory=list)
    null_counts: Dict[str, int] = field(default_factory=dict)
    row_count: int = 0
    date_range: Optional[tuple] = None
    errors: List[str] = field(default_factory=list)


def validate_revenue_trend_schema(df: pd.DataFrame) -> Schema65ValidationResult:
    """
    Validate that a DataFrame conforms to Schema 6.5 (Revenue Trend half).
    """
    errors = []
    missing_cols = [col for col in REVENUE_TREND_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if df.empty:
        errors.append("DataFrame is empty.")
        return Schema65ValidationResult(
            is_valid=False,
            missing_columns=missing_cols,
            errors=errors,
            row_count=0,
        )

    # Check date column
    if "period_date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["period_date"]):
            errors.append("'period_date' must be datetime dtype.")
    else:
        errors.append("Column 'period_date' is missing.")

    # Check is_forecast flag exists and boolean
    if "is_forecast" in df.columns and not pd.api.types.is_bool_dtype(df["is_forecast"]):
        # Allow numeric 0/1 or convert to bool
        pass

    null_counts = {col: int(df[col].isna().sum()) for col in df.columns if col in REVENUE_TREND_COLUMNS}
    date_range = (df["period_date"].min(), df["period_date"].max()) if "period_date" in df.columns else None

    return Schema65ValidationResult(
        is_valid=len(errors) == 0 and len(missing_cols) == 0,
        missing_columns=missing_cols,
        null_counts=null_counts,
        row_count=len(df),
        date_range=date_range,
        errors=errors,
    )


def validate_churn_trend_schema(df: pd.DataFrame) -> Schema65ValidationResult:
    """
    Validate that a DataFrame conforms to Schema 6.5 (Churn Trend half).
    """
    errors = []
    missing_cols = [col for col in CHURN_TREND_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if df.empty:
        errors.append("DataFrame is empty.")
        return Schema65ValidationResult(
            is_valid=False,
            missing_columns=missing_cols,
            errors=errors,
            row_count=0,
        )

    if "period_date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["period_date"]):
            errors.append("'period_date' must be datetime dtype.")
    else:
        errors.append("Column 'period_date' is missing.")

    # Churn rate and retention rate value sanity (between 0.0 and 1.0 or percent)
    if "churn_rate" in df.columns:
        invalid_rates = df[(df["churn_rate"] < 0) | (df["churn_rate"] > 1.0)]["churn_rate"]
        if not invalid_rates.empty:
            errors.append(f"Found {len(invalid_rates)} churn_rate values outside [0.0, 1.0].")

    null_counts = {col: int(df[col].isna().sum()) for col in df.columns if col in CHURN_TREND_COLUMNS}
    date_range = (df["period_date"].min(), df["period_date"].max()) if "period_date" in df.columns else None

    return Schema65ValidationResult(
        is_valid=len(errors) == 0 and len(missing_cols) == 0,
        missing_columns=missing_cols,
        null_counts=null_counts,
        row_count=len(df),
        date_range=date_range,
        errors=errors,
    )


def validate_combined_trend_schema(df: pd.DataFrame) -> Schema65ValidationResult:
    """
    Validate the combined master Schema 6.5 dataset.
    """
    errors = []
    missing_cols = [col for col in COMBINED_TREND_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if df.empty:
        errors.append("DataFrame is empty.")
        return Schema65ValidationResult(
            is_valid=False,
            missing_columns=missing_cols,
            errors=errors,
            row_count=0,
        )

    null_counts = {col: int(df[col].isna().sum()) for col in df.columns if col in COMBINED_TREND_COLUMNS}
    date_range = (df["period_date"].min(), df["period_date"].max()) if "period_date" in df.columns else None

    return Schema65ValidationResult(
        is_valid=len(errors) == 0 and len(missing_cols) == 0,
        missing_columns=missing_cols,
        null_counts=null_counts,
        row_count=len(df),
        date_range=date_range,
        errors=errors,
    )
