
import importlib
import sys
import types
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = "app.segmentation.customer_analytics.forecasting.forecasting"
OUTPUT_TABLES = ["forecast_revenue_trend", "forecast_churn_trend", "forecast_trend_combined"]


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def fake_engine():
    """Swap the real DB module for a stub so importing forecasting.py
    never reads .env or opens a MySQL connection."""
    importlib.import_module("app")           # keep real parent packages importable
    importlib.import_module("app.Database")

    engine = MagicMock(name="fake_engine")
    stub = types.ModuleType("app.Database.database")
    stub.engine = engine
    sys.modules["app.Database.database"] = stub
    sys.modules["app.Database"].database = stub
    return engine


@pytest.fixture(scope="session")
def fc(fake_engine):
    """The module under test, imported after the DB stub is in place."""
    return importlib.import_module(MODULE_PATH)


# ----------------------------------------------------------------------
# Test data builders
# ----------------------------------------------------------------------
def monthly_series(n=30, seed=0):
    """Trend + yearly seasonality + noise, monthly (MS) index."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 1000 + 10 * t + 50 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 5, n)
    return pd.Series(values, index=pd.date_range("2016-01-01", periods=n, freq="MS"))


def raw_trend_table(n=24, incomplete_last=0):
    """Rows shaped like historical_trend_combined (only the columns the code uses).
    `incomplete_last` = number of trailing partial periods (1 order each)."""
    df = pd.DataFrame({
        "granularity": "monthly",
        "period_date": pd.date_range("2016-09-01", periods=n, freq="MS"),
        "total_revenue": np.linspace(500_000, 1_000_000, n),
        "order_count": np.full(n, 6000),
        "churn_rate": np.r_[np.zeros(4), np.linspace(0.05, 0.12, n - 4)],  # early zero ramp-up
        "is_forecast": 0,
    })
    if incomplete_last:
        df.loc[df.index[-incomplete_last:], "order_count"] = 1  # partial periods at the end
    return df


# ----------------------------------------------------------------------
# logit / inv_logit
# ----------------------------------------------------------------------
class TestLogit:
    def test_round_trip(self, fc):
        p = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        np.testing.assert_allclose(fc.inv_logit(fc.logit(p)), p, atol=1e-6)

    def test_finite_at_0_and_1(self, fc):
        assert np.isfinite(fc.logit(0.0)) and np.isfinite(fc.logit(1.0))

    def test_inv_logit_bounded(self, fc):
        out = fc.inv_logit(np.array([-100, -1, 0, 1, 100]))
        assert ((out >= 0) & (out <= 1)).all()


# ----------------------------------------------------------------------
# trim_leading_zeros
# ----------------------------------------------------------------------
class TestTrimLeadingZeros:
    def test_trims_leading_run(self, fc):
        assert list(fc.trim_leading_zeros(pd.Series([0, 0, 0.05, 0.06]))) == [0.05, 0.06]

    def test_keeps_interior_zeros(self, fc):
        # Only the leading run is removed; later zeros are real observations.
        assert list(fc.trim_leading_zeros(pd.Series([0, 0.05, 0, 0.07]))) == [0.05, 0, 0.07]

    def test_no_leading_zeros_unchanged(self, fc):
        s = pd.Series([0.05, 0.06, 0.07])
        pd.testing.assert_series_equal(fc.trim_leading_zeros(s), s)

    def test_all_zeros_unchanged(self, fc):
        s = pd.Series([0.0, 0.0, 0.0])
        pd.testing.assert_series_equal(fc.trim_leading_zeros(s), s)

    def test_custom_threshold(self, fc):
        s = pd.Series([0.0001, 0.0002, 0.05, 0.06])
        assert list(fc.trim_leading_zeros(s, threshold=1e-3)) == [0.05, 0.06]


# ----------------------------------------------------------------------
# fill_zero_gaps
# ----------------------------------------------------------------------
class TestFillZeroGaps:
    def test_interpolates_interior_zeros(self, fc):
        s = pd.Series([0.04, 0.0, 0.0, 0.07])
        np.testing.assert_allclose(fc.fill_zero_gaps(s), [0.04, 0.05, 0.06, 0.07])

    def test_no_zeros_unchanged(self, fc):
        s = pd.Series([0.05, 0.06, 0.07])
        pd.testing.assert_series_equal(fc.fill_zero_gaps(s), s)

    def test_zero_at_end_filled_from_last_value(self, fc):
        assert list(fc.fill_zero_gaps(pd.Series([0.05, 0.06, 0.0]))) == [0.05, 0.06, 0.06]


# ----------------------------------------------------------------------
# fit_and_forecast
# ----------------------------------------------------------------------
class TestFitAndForecast:
    def test_horizon_and_columns(self, fc):
        summary, _ = fc.fit_and_forecast(monthly_series(30), horizon=6, freq="MS", season_len=12)
        assert len(summary) == 6
        assert {"mean", "pi_lower", "pi_upper"} <= set(summary.columns)

    def test_interval_ordered(self, fc):
        summary, _ = fc.fit_and_forecast(monthly_series(30), horizon=6, freq="MS", season_len=12)
        assert (summary["pi_lower"] <= summary["mean"]).all()
        assert (summary["mean"] <= summary["pi_upper"]).all()

    def test_forecast_dates_continue_after_history(self, fc):
        series = monthly_series(30)
        summary, _ = fc.fit_and_forecast(series, horizon=3, freq="MS", season_len=12)
        expected = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=3, freq="MS")
        assert list(summary.index) == list(expected)

    def test_seasonal_used_with_two_full_cycles(self, fc):
        _, label = fc.fit_and_forecast(monthly_series(24), horizon=3, freq="MS", season_len=12)
        assert "seasonal" in label

    def test_constant_series_gives_flat_forecast_without_warnings(self, fc):
        # Zero-variance input would make ETS take log(0); it must be handled explicitly.
        flat = pd.Series(0.0, index=pd.date_range("2016-01-01", periods=18, freq="MS"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            summary, label = fc.fit_and_forecast(flat, horizon=3, freq="MS", season_len=12)
        assert (summary[["mean", "pi_lower", "pi_upper"]] == 0).all().all()
        assert list(summary.index) == list(pd.date_range("2017-07-01", periods=3, freq="MS"))
        assert label.startswith("constant")

    def test_normal_series_fits_without_warnings(self, fc):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, label = fc.fit_and_forecast(monthly_series(30), horizon=6, freq="MS", season_len=12)
        assert "did not converge" not in label

    def test_falls_back_to_trend_only_with_short_history(self, fc):
        # < 2 seasonal cycles: ETS can't estimate seasonality, must not raise
        summary, label = fc.fit_and_forecast(monthly_series(15), horizon=3, freq="MS", season_len=12)
        assert len(summary) == 3
        assert "seasonal" not in label


# ----------------------------------------------------------------------
# evaluate_holdout
# ----------------------------------------------------------------------
class TestEvaluateHoldout:
    def test_metrics_finite_and_non_negative(self, fc):
        mape, rmse = fc.evaluate_holdout(monthly_series(30), 6, "MS", 12)
        assert np.isfinite(mape) and np.isfinite(rmse)
        assert mape >= 0 and rmse >= 0

    def test_zero_actuals_no_divide_by_zero(self, fc):
        # Training slice is 18 zeros (flat) and one holdout actual is 0.
        # Must produce finite metrics with no numerical warnings at all:
        # the flat series skips ETS, and MAPE skips the zero actual.
        values = np.r_[np.zeros(18), [0.0, 0.02, 0.03, 0.04, 0.05, 0.06]]
        series = pd.Series(values, index=pd.date_range("2016-01-01", periods=24, freq="MS"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            mape, rmse = fc.evaluate_holdout(series, 6, "MS", 12)
        assert mape == pytest.approx(100.0)  # flat 0 forecast vs non-zero actuals
        assert np.isfinite(rmse) and rmse >= 0

    def test_mape_ignores_zero_actuals(self, fc):
        # A zero in the holdout must not turn MAPE into inf/nan.
        values = np.r_[np.linspace(100, 200, 18), [0.0, 210, 220, 230, 240, 250]]
        series = pd.Series(values, index=pd.date_range("2016-01-01", periods=24, freq="MS"))
        mape, _ = fc.evaluate_holdout(series, 6, "MS", 12)
        assert np.isfinite(mape)

    def test_inverse_reports_metrics_in_rate_units(self, fc):
        # Churn is modelled in logit space; with inverse=inv_logit the RMSE
        # must be on the 0-1 rate scale, not logit units.
        rates = pd.Series(np.linspace(0.05, 0.12, 20),
                          index=pd.date_range("2017-01-01", periods=20, freq="MS"))
        _, rmse = fc.evaluate_holdout(rates.apply(fc.logit), 6, "MS", 12, inverse=fc.inv_logit)
        assert 0 <= rmse < 1


# ----------------------------------------------------------------------
# load_trend_data
# ----------------------------------------------------------------------
class TestLoadTrendData:
    def _load(self, fc, raw):
        with patch.object(fc.pd, "read_sql", return_value=raw) as mock_sql:
            return fc.load_trend_data(), mock_sql

    def test_indexed_by_period_date_with_revenue_column(self, fc):
        df, _ = self._load(fc, raw_trend_table(24))
        assert df.index.name == "period_date"
        assert df.index.is_monotonic_increasing
        assert "revenue" in df.columns and "total_revenue" not in df.columns

    def test_queries_configured_granularity(self, fc):
        _, mock_sql = self._load(fc, raw_trend_table(24))
        assert mock_sql.call_args.kwargs["params"] == {"granularity": fc.GRANULARITY}

    def test_drops_incomplete_last_period(self, fc):
        df, _ = self._load(fc, raw_trend_table(24, incomplete_last=1))
        assert len(df) == 23

    def test_drops_several_incomplete_trailing_periods(self, fc):
        # Real data ends with two partial months (Sep and Oct 2018); both must go.
        df, _ = self._load(fc, raw_trend_table(24, incomplete_last=2))
        assert len(df) == 22
        assert df["order_count"].iloc[-1] == 6000

    def test_keeps_normal_last_period(self, fc):
        df, _ = self._load(fc, raw_trend_table(24))
        assert len(df) == 24

    def test_empty_result_raises(self, fc):
        empty = raw_trend_table(24).iloc[0:0]
        with pytest.raises(ValueError):
            self._load(fc, empty)


# ----------------------------------------------------------------------
# main() end-to-end (mocked DB, CSVs go to a temp folder)
# ----------------------------------------------------------------------
class TestMain:
    def _run(self, fc, tmp_path, monkeypatch):
        monkeypatch.setattr(fc, "OUTPUT_DIR", tmp_path)
        with patch.object(fc.pd, "read_sql", return_value=raw_trend_table(30)), \
             patch.object(pd.DataFrame, "to_sql", autospec=True) as mock_to_sql:
            fc.main()
        # autospec passes the DataFrame as args[0] and the table name as args[1]
        return {c.args[1]: c.args[0] for c in mock_to_sql.call_args_list}

    def test_writes_all_three_tables_and_csvs(self, fc, tmp_path, monkeypatch):
        written = self._run(fc, tmp_path, monkeypatch)
        assert sorted(written) == sorted(OUTPUT_TABLES)
        for table in OUTPUT_TABLES:
            assert (tmp_path / f"{table}.csv").exists()

    def test_output_shape_and_values(self, fc, tmp_path, monkeypatch):
        written = self._run(fc, tmp_path, monkeypatch)
        combined = written["forecast_trend_combined"]

        assert len(combined) == fc.FORECAST_HORIZON
        assert {"period_date", "granularity", "generated_at",
                "forecasted_revenue", "forecasted_churn_rate"} <= set(combined.columns)
        # Forecast starts the month after the last historical period (2016-09 + 30 months)
        assert combined["period_date"].iloc[0] == pd.Timestamp("2019-03-01")
        # Churn forecast and its interval must stay valid rates
        for col in ("forecasted_churn_rate", "ci_lower_churn", "ci_upper_churn"):
            assert combined[col].between(0, 1).all()
        # Revenue and its interval can never be negative
        for col in ("forecasted_revenue", "ci_lower_revenue", "ci_upper_revenue"):
            assert (combined[col] >= 0).all()