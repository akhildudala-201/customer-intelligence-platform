"""
Unit tests for:
    app/segmentation/customer_analytics/forecasting/forecasting.py

Run from the project root with:
    pytest app/segmentation/customer_analytics/forecasting/test_forecasting.py -v

Or, if placed elsewhere, adjust MODULE_PATH below to match where
forecasting.py actually lives as an importable module.

WHY THE STUBBING:
forecasting.py does `from app.Database.database import engine` at import
time. That module reads a real .env file and raises if DB_HOST etc. are
missing. To keep these tests fast, deterministic, and runnable without a
live DB or .env file, we inject a fake app.Database.database module into
sys.modules BEFORE forecasting.py is imported, so the import line
succeeds against a MagicMock engine instead of a real connection.
"""

import sys
import types
import importlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Change this if your test file lives somewhere else relative to the
# project root / your import path.
MODULE_PATH = "app.segmentation.customer_analytics.forecasting.forecasting"


# ----------------------------------------------------------------------
# Stub app.Database.database BEFORE forecasting.py is imported.
#
# IMPORTANT: we do NOT replace `app` or `app.Database` themselves with
# fake modules -- doing so strips their real __path__ and breaks every
# other import under app.* (e.g. app.segmentation...forecasting), which
# is what caused "ModuleNotFoundError: No module named 'app.segmentation';
# 'app' is not a package" the first time around. Instead we let the real
# `app` and `app.Database` packages import normally, and only replace the
# leaf module `app.Database.database` with a fake one.
# ----------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _stub_database_module():
    # Import the REAL parent packages first, so their __path__ stays intact
    # and other app.* submodules remain importable normally.
    importlib.import_module("app")
    importlib.import_module("app.Database")

    fake_engine = MagicMock(name="fake_engine")

    fake_db_module = types.ModuleType("app.Database.database")
    fake_db_module.engine = fake_engine

    sys.modules["app.Database.database"] = fake_db_module
    # Keep dotted attribute access consistent too (app.Database.database)
    sys.modules["app.Database"].database = fake_db_module

    yield fake_engine


@pytest.fixture(scope="session")
def forecasting(_stub_database_module):
    """Import the module under test after the DB stub is in place."""
    module = importlib.import_module(MODULE_PATH)
    return module


# ----------------------------------------------------------------------
# logit / inv_logit
# ----------------------------------------------------------------------
class TestLogitInvLogit:
    def test_round_trip(self, forecasting):
        p = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        recovered = forecasting.inv_logit(forecasting.logit(p))
        np.testing.assert_allclose(recovered, p, atol=1e-6)

    def test_handles_0_and_1_without_error(self, forecasting):
        assert np.isfinite(forecasting.logit(0.0))
        assert np.isfinite(forecasting.logit(1.0))

    def test_inv_logit_stays_within_0_and_1(self, forecasting):
        x = np.array([-100, -1, 0, 1, 100])
        result = forecasting.inv_logit(x)
        assert np.all(result >= 0) and np.all(result <= 1)


# ----------------------------------------------------------------------
# trim_leading_zeros
# ----------------------------------------------------------------------
class TestTrimLeadingZeros:
    def test_trims_leading_zero_run(self, forecasting):
        s = pd.Series([0, 0, 0, 0.05, 0.06, 0.07])
        trimmed = forecasting.trim_leading_zeros(s)
        assert list(trimmed.values) == [0.05, 0.06, 0.07]

    def test_no_leading_zeros_returns_unchanged(self, forecasting):
        s = pd.Series([0.05, 0.06, 0.07])
        trimmed = forecasting.trim_leading_zeros(s)
        pd.testing.assert_series_equal(trimmed, s)

    def test_all_zeros_returns_unchanged(self, forecasting):
        s = pd.Series([0, 0, 0])
        trimmed = forecasting.trim_leading_zeros(s)
        pd.testing.assert_series_equal(trimmed, s)

    def test_respects_custom_threshold(self, forecasting):
        s = pd.Series([0.0001, 0.0002, 0.05, 0.06])
        trimmed = forecasting.trim_leading_zeros(s, threshold=1e-3)
        assert list(trimmed.values) == [0.05, 0.06]


# ----------------------------------------------------------------------
# fit_and_forecast (ETS)
# ----------------------------------------------------------------------
def _synthetic_monthly_series(n=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=n, freq="MS")
    seasonal = 50 * np.sin(np.arange(n) * (2 * np.pi / 12))
    trend = np.arange(n) * 10
    noise = rng.normal(0, 5, n)
    values = 1000 + trend + seasonal + noise
    return pd.Series(values, index=dates)


class TestFitAndForecast:
    def test_returns_expected_horizon_and_columns(self, forecasting):
        series = _synthetic_monthly_series(n=30)
        _, summary = forecasting.fit_and_forecast(series, horizon=6, seasonal_periods=12)
        assert len(summary) == 6
        for col in ("mean", "pi_lower", "pi_upper"):
            assert col in summary.columns

    def test_confidence_interval_bounds_are_ordered(self, forecasting):
        series = _synthetic_monthly_series(n=30)
        _, summary = forecasting.fit_and_forecast(series, horizon=6, seasonal_periods=12)
        assert (summary["pi_lower"] <= summary["mean"]).all()
        assert (summary["mean"] <= summary["pi_upper"]).all()

    def test_does_not_crash_with_short_history(self, forecasting):
        # Fewer than 2 full seasonal cycles (< 24 points for period=12)
        # should trigger the non-seasonal fallback rather than raising.
        series = _synthetic_monthly_series(n=15)
        _, summary = forecasting.fit_and_forecast(series, horizon=3, seasonal_periods=12)
        assert len(summary) == 3


# ----------------------------------------------------------------------
# evaluate_holdout
# ----------------------------------------------------------------------
class TestEvaluateHoldout:
    def test_returns_finite_non_negative_metrics(self, forecasting):
        series = _synthetic_monthly_series(n=30)
        mape, rmse = forecasting.evaluate_holdout(series, horizon=6, seasonal_periods=12)
        assert np.isfinite(mape)
        assert np.isfinite(rmse)
        assert mape >= 0
        assert rmse >= 0

    def test_handles_near_zero_actuals_without_crashing(self, forecasting):
        dates = pd.date_range("2016-01-01", periods=24, freq="MS")
        values = np.concatenate(
            [np.zeros(18), np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])]
        )
        series = pd.Series(values, index=dates)
        # Should not raise ZeroDivisionError even with near-zero actuals
        mape, rmse = forecasting.evaluate_holdout(series, horizon=6, seasonal_periods=12)
        assert rmse >= 0


# ----------------------------------------------------------------------
# load_trend_data
# ----------------------------------------------------------------------
class TestLoadTrendData:
    @staticmethod
    def _make_raw_df(n=24, incomplete_trailing_period=False):
        dates = pd.date_range("2016-09-01", periods=n, freq="MS")
        df = pd.DataFrame({
            "granularity": "monthly",
            "period_date": dates,
            "total_revenue": np.linspace(500_000, 1_000_000, n),
            "order_count": np.full(n, 6000),
            "churn_rate": np.linspace(0.0, 0.12, n),
            "is_forecast": 0,
        })
        if incomplete_trailing_period:
            df.loc[df.index[-1], "order_count"] = 1  # simulate mid-month partial pull
        return df

    def test_renames_columns_and_sorts_by_date(self, forecasting):
        raw = self._make_raw_df(n=24)
        with patch.object(forecasting.pd, "read_sql", return_value=raw):
            df = forecasting.load_trend_data(forecasting.engine)
        assert "period_start" in df.columns
        assert "revenue" in df.columns
        assert df["period_start"].is_monotonic_increasing

    def test_drops_incomplete_trailing_period(self, forecasting):
        raw = self._make_raw_df(n=24, incomplete_trailing_period=True)
        with patch.object(forecasting.pd, "read_sql", return_value=raw):
            df = forecasting.load_trend_data(forecasting.engine)
        # last row should have been dropped as "likely incomplete"
        assert len(df) == 23

    def test_keeps_full_data_when_trailing_period_looks_normal(self, forecasting):
        raw = self._make_raw_df(n=24, incomplete_trailing_period=False)
        with patch.object(forecasting.pd, "read_sql", return_value=raw):
            df = forecasting.load_trend_data(forecasting.engine)
        assert len(df) == 24

    def test_raises_value_error_on_empty_result(self, forecasting):
        empty = pd.DataFrame(
            columns=["granularity", "period_date", "total_revenue", "order_count", "churn_rate"]
        )
        with patch.object(forecasting.pd, "read_sql", return_value=empty):
            with pytest.raises(ValueError):
                forecasting.load_trend_data(forecasting.engine)


# ----------------------------------------------------------------------
# write_forecast_to_mysql
# ----------------------------------------------------------------------
class TestWriteForecastToMysql:
    def test_writes_all_three_tables_with_correct_names(self, forecasting):
        revenue_df = MagicMock()
        churn_df = MagicMock()
        combined_df = MagicMock()
        fake_engine = MagicMock()

        forecasting.write_forecast_to_mysql(fake_engine, revenue_df, churn_df, combined_df)

        revenue_df.to_sql.assert_called_once_with(
            "forecast_revenue_trend", fake_engine, if_exists="replace", index=False
        )
        churn_df.to_sql.assert_called_once_with(
            "forecast_churn_trend", fake_engine, if_exists="replace", index=False
        )
        combined_df.to_sql.assert_called_once_with(
            "forecast_trend_combined", fake_engine, if_exists="replace", index=False
        )


# ----------------------------------------------------------------------
# End-to-end smoke test for main(), fully mocked (no real DB writes,
# no real network calls) -- just checks the pipeline runs top to bottom
# without raising, given clean synthetic input data.
# ----------------------------------------------------------------------
class TestMainSmokeTest:
    def test_main_runs_without_raising(self, forecasting, tmp_path, monkeypatch):
        raw = TestLoadTrendData._make_raw_df(n=30, incomplete_trailing_period=False)

        # Route local CSV backups to a temp dir instead of the real project folder
        monkeypatch.setattr(forecasting, "OUTPUT_DIR", tmp_path)

        with patch.object(forecasting.pd, "read_sql", return_value=raw), \
             patch.object(forecasting, "write_forecast_to_mysql") as mock_write:
            forecasting.main()

        # write_forecast_to_mysql should have been called exactly once
        mock_write.assert_called_once()

        # And local CSV backups should exist
        assert (tmp_path / "forecast_revenue_trend.csv").exists()
        assert (tmp_path / "forecast_churn_trend.csv").exists()
        assert (tmp_path / "forecast_trend_combined.csv").exists()