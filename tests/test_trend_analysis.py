"""
tests/test_trend_analysis.py

Unit and integration tests for Person 5 — Historical Trend Analysis (Schema 6.5).
Tests Daily, Weekly, and Monthly Revenue & Churn trend engines, schema validation,
gap-filling, moving averages, data loader, and reporting.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
for path in (str(PROJECT_ROOT), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from app.Segmentation.customer_analytics.Trends.data_loader import TrendDataLoader
from app.Segmentation.customer_analytics.Trends.export import TrendExporter
from app.Segmentation.customer_analytics.Trends.schemas import (
    CHURN_TREND_COLUMNS,
    COMBINED_TREND_COLUMNS,
    REVENUE_TREND_COLUMNS,
    ChurnTrendRecord,
    RevenueTrendRecord,
    validate_churn_trend_schema,
    validate_combined_trend_schema,
    validate_revenue_trend_schema,
)
from app.Segmentation.customer_analytics.Trends.trend_engine import HistoricalTrendEngine


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_orders_df():
    """Generates a synthetic order dataset spanning multiple months."""
    data = [
        # Customer 1: 3 orders across months
        {"order_id": "o1", "customer_unique_id": "c1", "order_purchase_timestamp": "2017-01-05 10:00:00", "order_value": 100.0, "order_status": "delivered"},
        {"order_id": "o2", "customer_unique_id": "c1", "order_purchase_timestamp": "2017-02-10 12:00:00", "order_value": 150.0, "order_status": "delivered"},
        {"order_id": "o3", "customer_unique_id": "c1", "order_purchase_timestamp": "2017-04-15 14:00:00", "order_value": 200.0, "order_status": "delivered"},
        # Customer 2: 1 order
        {"order_id": "o4", "customer_unique_id": "c2", "order_purchase_timestamp": "2017-01-20 09:00:00", "order_value": 50.0, "order_status": "delivered"},
        # Customer 3: 2 orders on same day
        {"order_id": "o5", "customer_unique_id": "c3", "order_purchase_timestamp": "2017-03-01 11:00:00", "order_value": 80.0, "order_status": "delivered"},
        {"order_id": "o6", "customer_unique_id": "c3", "order_purchase_timestamp": "2017-03-01 16:00:00", "order_value": 120.0, "order_status": "delivered"},
    ]
    df = pd.DataFrame(data)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return df


# =============================================================================
# Revenue Trend Tests
# =============================================================================

def test_daily_revenue_trend_schema_and_values(sample_orders_df):
    engine = HistoricalTrendEngine()
    df_daily = engine.compute_revenue_trend(sample_orders_df, granularity="daily")

    # Schema checks
    val = validate_revenue_trend_schema(df_daily)
    assert val.is_valid, f"Validation failed: {val.errors}"
    assert list(df_daily.columns) == REVENUE_TREND_COLUMNS

    # Metadata checks
    assert (df_daily["granularity"] == "daily").all()
    assert (~df_daily["is_forecast"]).all()

    # Gap-filling check (continuous days between 2017-01-05 and 2017-04-15)
    min_date = df_daily["period_date"].min()
    max_date = df_daily["period_date"].max()
    expected_days = (max_date - min_date).days + 1
    assert len(df_daily) == expected_days

    # Value checks on specific active day (2017-03-01: o5 + o6 = 200.0)
    day_row = df_daily[df_daily["period_date"] == pd.Timestamp("2017-03-01")].iloc[0]
    assert day_row["total_revenue"] == 200.0
    assert day_row["order_count"] == 2
    assert day_row["unique_customers"] == 1
    assert day_row["avg_order_value"] == 100.0
    assert day_row["avg_revenue_per_user"] == 200.0


def test_weekly_revenue_trend(sample_orders_df):
    engine = HistoricalTrendEngine()
    df_weekly = engine.compute_revenue_trend(sample_orders_df, granularity="weekly")

    val = validate_revenue_trend_schema(df_weekly)
    assert val.is_valid, f"Validation failed: {val.errors}"
    assert (df_weekly["granularity"] == "weekly").all()
    assert df_weekly["total_revenue"].sum() == pytest.approx(700.0)
    assert df_weekly["cumulative_revenue"].iloc[-1] == pytest.approx(700.0)


def test_monthly_revenue_trend(sample_orders_df):
    engine = HistoricalTrendEngine()
    df_monthly = engine.compute_revenue_trend(sample_orders_df, granularity="monthly")

    val = validate_revenue_trend_schema(df_monthly)
    assert val.is_valid, f"Validation failed: {val.errors}"
    assert (df_monthly["granularity"] == "monthly").all()

    # Dates should be 1st of month
    assert (df_monthly["period_date"].dt.day == 1).all()

    # Total sum check across all months
    assert df_monthly["total_revenue"].sum() == pytest.approx(700.0)
    assert df_monthly["order_count"].sum() == 6


# =============================================================================
# Churn Trend Tests
# =============================================================================

def test_churn_trend_monthly(sample_orders_df):
    engine = HistoricalTrendEngine(return_window_days=60)
    df_churn = engine.compute_churn_trend(sample_orders_df, granularity="monthly")

    val = validate_churn_trend_schema(df_churn)
    assert val.is_valid, f"Validation failed: {val.errors}"
    assert list(df_churn.columns) == CHURN_TREND_COLUMNS
    assert (df_churn["granularity"] == "monthly").all()

    # Churn rates must be within valid probabilities [0.0, 1.0]
    assert (df_churn["churn_rate"] >= 0.0).all()
    assert (df_churn["churn_rate"] <= 1.0).all()
    assert (df_churn["retention_rate"] >= 0.0).all()
    assert (df_churn["retention_rate"] <= 1.0).all()
    assert np.allclose(df_churn["churn_rate"] + df_churn["retention_rate"], 1.0)


def test_combined_trend_master_schema(sample_orders_df):
    engine = HistoricalTrendEngine()
    results = engine.generate_all_trends(sample_orders_df)

    assert "daily" in results
    assert "weekly" in results
    assert "monthly" in results
    assert "all_granularities_combined" in results

    master_df = results["all_granularities_combined"]["master"]
    val = validate_combined_trend_schema(master_df)
    assert val.is_valid, f"Master Schema 6.5 validation failed: {val.errors}"
    assert list(master_df.columns) == COMBINED_TREND_COLUMNS


# =============================================================================
# Edge Cases & Error Handling Tests
# =============================================================================

def test_empty_dataframe_handling():
    engine = HistoricalTrendEngine()
    empty_orders = pd.DataFrame(columns=["order_id", "customer_unique_id", "order_purchase_timestamp", "order_value"])

    rev_df = engine.compute_revenue_trend(empty_orders, granularity="daily")
    churn_df = engine.compute_churn_trend(empty_orders, granularity="daily")
    combined_df = engine.build_combined_trend(rev_df, churn_df)

    assert rev_df.empty
    assert churn_df.empty
    assert combined_df.empty
    assert list(rev_df.columns) == REVENUE_TREND_COLUMNS
    assert list(churn_df.columns) == CHURN_TREND_COLUMNS
    assert list(combined_df.columns) == COMBINED_TREND_COLUMNS


def test_schema_validators_reject_invalid_data():
    invalid_rev_df = pd.DataFrame({
        "granularity": ["daily"],
        "total_revenue": [100.0],
        # Missing required period_date, is_forecast, etc.
    })
    val = validate_revenue_trend_schema(invalid_rev_df)
    assert not val.is_valid
    assert len(val.missing_columns) > 0


def test_exporter_creates_files(sample_orders_df, tmp_path):
    engine = HistoricalTrendEngine()
    results = engine.generate_all_trends(sample_orders_df)

    exporter = TrendExporter(output_dir=tmp_path)
    files = exporter.export_reports(results)

    assert (tmp_path / "historical_revenue_trend_daily.csv").exists()
    assert (tmp_path / "historical_churn_trend_monthly.csv").exists()
    assert (tmp_path / "historical_trend_master_schema_6_5.csv").exists()
    assert (tmp_path / "trend_summary_report.json").exists()


def test_data_loader_config_and_initialization():
    loader = TrendDataLoader()
    assert loader.config is not None
    assert "reference_date" in loader.config
    assert "return_window_days" in loader.config
