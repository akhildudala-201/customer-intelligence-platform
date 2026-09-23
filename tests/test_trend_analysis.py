"""
tests/test_trend_analysis.py

Comprehensive unit and integration tests for Person 5 — Historical Trend Analysis (Schema 6.5).
Tests Daily, Weekly, and Monthly Revenue & Churn trend engines, right-censoring handling,
survivorship-based active customer denominator, date gap-filling, schema validation,
empty DataFrame edge cases, and file export.
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

from app.segmentation.customer_analytics.Trends.data_loader import TrendDataLoader
from app.segmentation.customer_analytics.Trends.export import TrendExporter
from app.segmentation.customer_analytics.Trends.schemas import (
    CHURN_TREND_COLUMNS,
    COMBINED_TREND_COLUMNS,
    REVENUE_TREND_COLUMNS,
    ChurnTrendRecord,
    RevenueTrendRecord,
    validate_churn_trend_schema,
    validate_combined_trend_schema,
    validate_revenue_trend_schema,
)
from app.segmentation.customer_analytics.Trends.trend_engine import HistoricalTrendEngine


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


@pytest.fixture
def sample_churn_labels_df():
    """Generates customer churn labels corresponding to the sample orders."""
    # Reference date: 2017-10-17, return_window: 60 days
    # c1: last order 2017-04-15 -> days_since = 185 days -> repeat on o2, o3 -> but after 2017-04-15 no return -> churn at 2017-06-14 (label=1)
    # c2: last order 2017-01-20 -> days_since = 270 days -> no repeat -> churn at 2017-03-21 (label=1)
    # c3: last order 2017-03-01 -> days_since = 230 days -> no repeat after day 1 -> churn at 2017-04-30 (label=1)
    return pd.DataFrame([
        {
            "customer_unique_id": "c1",
            "first_order_date": pd.Timestamp("2017-01-05"),
            "last_order_date": pd.Timestamp("2017-04-15"),
            "days_since_last_order": 185.0,
            "num_valid_orders": 3,
            "censored": False,
            "label": 1,
        },
        {
            "customer_unique_id": "c2",
            "first_order_date": pd.Timestamp("2017-01-20"),
            "last_order_date": pd.Timestamp("2017-01-20"),
            "days_since_last_order": 270.0,
            "num_valid_orders": 1,
            "censored": False,
            "label": 1,
        },
        {
            "customer_unique_id": "c3",
            "first_order_date": pd.Timestamp("2017-03-01"),
            "last_order_date": pd.Timestamp("2017-03-01"),
            "days_since_last_order": 230.0,
            "num_valid_orders": 2,
            "censored": False,
            "label": 1,
        },
    ])


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
# Churn Trend & Survivorship Denominator Tests (Issue 1, 2, 3 Fixed)
# =============================================================================

def test_churn_trend_consumes_labels_and_computes_survivorship(sample_orders_df, sample_churn_labels_df):
    engine = HistoricalTrendEngine(return_window_days=60, reference_date="2017-10-17")
    df_churn = engine.compute_churn_trend(
        sample_orders_df,
        churn_labels_df=sample_churn_labels_df,
        granularity="monthly",
    )

    val = validate_churn_trend_schema(df_churn)
    assert val.is_valid, f"Validation failed: {val.errors}"
    assert list(df_churn.columns) == CHURN_TREND_COLUMNS
    assert (df_churn["granularity"] == "monthly").all()

    # Total churned customers across all periods must match labels with label == 1 (3 customers)
    assert df_churn["churned_customers"].sum() == 3

    # Churn rates must be within valid bounds [0.0, 1.0]
    assert (df_churn["churn_rate"] >= 0.0).all()
    assert (df_churn["churn_rate"] <= 1.0).all()
    assert (df_churn["retention_rate"] >= 0.0).all()
    assert (df_churn["retention_rate"] <= 1.0).all()
    assert np.allclose(df_churn["churn_rate"] + df_churn["retention_rate"], 1.0)

    # Active customers should reflect surviving customer base
    assert (df_churn["active_customers"] >= 0).all()


def test_right_censoring_tail_flag(sample_orders_df):
    """
    Test Issue 1: Periods within return_window_days of reference_date
    are marked with is_censored = True so Person 6 knows observation is incomplete.
    """
    ref_date = pd.Timestamp("2017-06-01")
    return_window = 60  # Censoring threshold is 2017-04-02
    engine = HistoricalTrendEngine(return_window_days=return_window, reference_date=ref_date)

    df_churn = engine.compute_churn_trend(
        sample_orders_df,
        granularity="monthly",
        reference_date=ref_date,
    )

    # Periods: 2017-01-01, 2017-02-01, 2017-03-01 -> before 2017-04-02 -> is_censored = False
    # Period: 2017-04-01, 2017-05-01 -> is_censored = True (tail)
    assert not df_churn[df_churn["period_date"] == pd.Timestamp("2017-01-01")]["is_censored"].iloc[0]
    assert df_churn[df_churn["period_date"] >= pd.Timestamp("2017-05-01")]["is_censored"].iloc[0]


# =============================================================================
# Gap-Filling on Sparse Dates Tests (Issue 6)
# =============================================================================

def test_gap_filling_sparse_dates():
    """
    Verify continuous date index with no date gaps across daily, weekly, and monthly.
    """
    sparse_orders = pd.DataFrame([
        {"order_id": "o1", "customer_unique_id": "c1", "order_purchase_timestamp": "2017-01-01 10:00:00", "order_value": 100.0, "order_status": "delivered"},
        {"order_id": "o2", "customer_unique_id": "c2", "order_purchase_timestamp": "2017-01-10 10:00:00", "order_value": 200.0, "order_status": "delivered"},
    ])
    engine = HistoricalTrendEngine()

    # Daily: exactly 10 days (Jan 1 to Jan 10)
    df_daily = engine.compute_revenue_trend(sparse_orders, granularity="daily")
    assert len(df_daily) == 10
    assert df_daily.iloc[1]["total_revenue"] == 0.0  # Jan 2 has 0 revenue, no NaN
    assert df_daily.iloc[1]["order_count"] == 0

    # Monthly: Jan 2017
    df_monthly = engine.compute_revenue_trend(sparse_orders, granularity="monthly")
    assert len(df_monthly) == 1
    assert df_monthly.iloc[0]["total_revenue"] == 300.0


# =============================================================================
# Schema Conformance on Empty DataFrames (Issue 6)
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


# =============================================================================
# Schema 6.5 Validation Enforcement (Issue 4)
# =============================================================================

def test_schema_validators_reject_invalid_data():
    invalid_rev_df = pd.DataFrame({
        "granularity": ["daily"],
        "total_revenue": [100.0],
        # Missing required period_date, is_forecast, etc.
    })
    val = validate_revenue_trend_schema(invalid_rev_df)
    assert not val.is_valid
    assert len(val.missing_columns) > 0


def test_generate_all_trends_validation_enforcement(sample_orders_df):
    engine = HistoricalTrendEngine()
    results = engine.generate_all_trends(sample_orders_df, raise_on_validation_error=True)

    assert "daily" in results
    assert "weekly" in results
    assert "monthly" in results
    assert "all_granularities_combined" in results

    # Verify validation reports exist and all are valid
    val_report = results["all_granularities_combined"]["validation"]
    for gran in ("daily", "weekly", "monthly"):
        assert val_report[gran]["revenue"]["is_valid"]
        assert val_report[gran]["churn"]["is_valid"]
        assert val_report[gran]["combined"]["is_valid"]

    master_df = results["all_granularities_combined"]["master"]
    val_master = validate_combined_trend_schema(master_df)
    assert val_master.is_valid, f"Master Schema 6.5 validation failed: {val_master.errors}"
    assert list(master_df.columns) == COMBINED_TREND_COLUMNS


# =============================================================================
# Data Loader and Exporter Tests
# =============================================================================

def test_data_loader_config_and_initialization():
    loader = TrendDataLoader()
    assert loader.config is not None
    assert "reference_date" in loader.config
    assert "return_window_days" in loader.config


def test_exporter_creates_files(sample_orders_df, tmp_path):
    engine = HistoricalTrendEngine()
    results = engine.generate_all_trends(sample_orders_df)

    exporter = TrendExporter(output_dir=tmp_path)
    files = exporter.export_reports(results)

    assert (tmp_path / "historical_revenue_trend_daily.csv").exists()
    assert (tmp_path / "historical_churn_trend_monthly.csv").exists()
    assert (tmp_path / "historical_trend_master_schema_6_5.csv").exists()
    assert (tmp_path / "trend_summary_report.json").exists()
