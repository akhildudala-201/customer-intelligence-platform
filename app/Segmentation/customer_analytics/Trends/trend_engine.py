"""
app/Trends/trend_engine.py

Core Historical Trend Analysis Engine (Person 5).
Produces aggregated, continuous historical time-series trends for Churn and Revenue
at Daily, Weekly, and Monthly levels, conforming to Schema 6.5 (trend half).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    from app.Segmentation.customer_analytics.Trends.schemas import (
        CHURN_TREND_COLUMNS,
        COMBINED_TREND_COLUMNS,
        REVENUE_TREND_COLUMNS,
        validate_churn_trend_schema,
        validate_combined_trend_schema,
        validate_revenue_trend_schema,
    )
except (ImportError, ModuleNotFoundError):
    from .schemas import (
        CHURN_TREND_COLUMNS,
        COMBINED_TREND_COLUMNS,
        REVENUE_TREND_COLUMNS,
        validate_churn_trend_schema,
        validate_combined_trend_schema,
        validate_revenue_trend_schema,
    )


class HistoricalTrendEngine:
    """
    Computes time-series trend aggregations for e-commerce transactions and customer churn.
    """

    SUPPORTED_GRANULARITIES = ("daily", "weekly", "monthly")

    def __init__(
        self,
        return_window_days: int = 180,
        rolling_windows: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        self.return_window_days = return_window_days
        self.rolling_windows = rolling_windows or {
            "daily": {"short": 7, "long": 30},
            "weekly": {"short": 4, "long": 12},
            "monthly": {"short": 3, "long": 6},
        }

    # =========================================================================
    # Helper: Granularity Period Normalization & Resampling
    # =========================================================================

    def _get_freq_alias(self, granularity: str) -> str:
        """Map granularity name to pandas frequency string."""
        granularity = granularity.lower()
        if granularity in ("daily", "d", "day"):
            return "D"
        elif granularity in ("weekly", "w", "week"):
            return "W-MON"
        elif granularity in ("monthly", "m", "month"):
            return "MS"
        else:
            raise ValueError(
                f"Unsupported granularity: '{granularity}'. "
                f"Must be one of {self.SUPPORTED_GRANULARITIES}"
            )

    def _normalize_timestamp_to_period(
        self, series: pd.Series, granularity: str
    ) -> pd.Series:
        """
        Normalize timestamps to the period start date (Timestamp).
        Daily: YYYY-MM-DD 00:00:00
        Weekly: Monday of the week 00:00:00
        Monthly: 1st of the month 00:00:00
        """
        dt_series = pd.to_datetime(series)
        gran = granularity.lower()
        if gran in ("daily", "d", "day"):
            return dt_series.dt.floor("D")
        elif gran in ("weekly", "w", "week"):
            # Start of week (Monday)
            return dt_series.dt.to_period("W-SUN").dt.start_time
        elif gran in ("monthly", "m", "month"):
            # Start of month
            return dt_series.dt.to_period("M").dt.start_time
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")

    # =========================================================================
    # Revenue Trend Computation
    # =========================================================================

    def compute_revenue_trend(
        self,
        orders_df: pd.DataFrame,
        granularity: str = "daily",
        min_date: Optional[pd.Timestamp] = None,
        max_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Compute historical revenue trend for a specified granularity.

        Parameters:
            orders_df: DataFrame containing ['order_id', 'customer_unique_id',
                       'order_purchase_timestamp', 'order_value']
            granularity: 'daily', 'weekly', or 'monthly'
            min_date: Optional lower bound date
            max_date: Optional upper bound date

        Returns:
            DataFrame adhering to Schema 6.5 (Revenue Trend half).
        """
        if orders_df.empty:
            empty_df = pd.DataFrame(columns=REVENUE_TREND_COLUMNS)
            empty_df["period_date"] = pd.to_datetime(empty_df["period_date"])
            empty_df["is_forecast"] = empty_df["is_forecast"].astype(bool)
            return empty_df

        df = orders_df.copy()
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
        df["order_value"] = pd.to_numeric(df["order_value"], errors="coerce").fillna(0.0)

        # Period start normalization
        df["period_date"] = self._normalize_timestamp_to_period(
            df["order_purchase_timestamp"], granularity
        )

        # Aggregate raw metrics by period
        grouped = df.groupby("period_date").agg(
            total_revenue=("order_value", "sum"),
            order_count=("order_id", "nunique"),
            unique_customers=("customer_unique_id", "nunique"),
        )

        # Continuous date range gap-filling (vital for time-series forecasting)
        freq = self._get_freq_alias(granularity)
        start_date = min_date or df["period_date"].min()
        end_date = max_date or df["period_date"].max()

        full_index = pd.date_range(start=start_date, end=end_date, freq=freq)
        resampled = grouped.reindex(full_index)

        resampled["total_revenue"] = resampled["total_revenue"].fillna(0.0)
        resampled["order_count"] = resampled["order_count"].fillna(0).astype(int)
        resampled["unique_customers"] = resampled["unique_customers"].fillna(0).astype(int)

        # Derived unit economics
        resampled["avg_order_value"] = np.where(
            resampled["order_count"] > 0,
            resampled["total_revenue"] / resampled["order_count"],
            0.0,
        )
        resampled["avg_revenue_per_user"] = np.where(
            resampled["unique_customers"] > 0,
            resampled["total_revenue"] / resampled["unique_customers"],
            0.0,
        )

        # Growth Rate (% change from previous period)
        prev_rev = resampled["total_revenue"].shift(1)
        resampled["revenue_growth_pct"] = np.where(
            (prev_rev > 0) & (prev_rev.notna()),
            ((resampled["total_revenue"] - prev_rev) / prev_rev) * 100.0,
            np.where(
                (prev_rev == 0) & (resampled["total_revenue"] > 0),
                100.0,
                0.0,
            ),
        )
        # First period has NA/0 growth
        resampled.iloc[0, resampled.columns.get_loc("revenue_growth_pct")] = None

        # Rolling Moving Averages
        windows = self.rolling_windows.get(granularity.lower(), {"short": 7, "long": 30})
        short_w = windows.get("short", 7)
        long_w = windows.get("long", 30)

        resampled["rolling_revenue_sma_short"] = (
            resampled["total_revenue"].rolling(window=short_w, min_periods=1).mean()
        )
        resampled["rolling_revenue_sma_long"] = (
            resampled["total_revenue"].rolling(window=long_w, min_periods=1).mean()
        )

        # Cumulative Revenue
        resampled["cumulative_revenue"] = resampled["total_revenue"].cumsum()

        # Reset index to column and append metadata
        resampled = resampled.reset_index().rename(columns={"index": "period_date"})
        resampled["granularity"] = granularity.lower()
        resampled["is_forecast"] = False

        # Reorder to exact Schema 6.5 contract
        result = resampled[REVENUE_TREND_COLUMNS].copy()

        # Ensure float rounding for clean precision
        result["total_revenue"] = result["total_revenue"].round(2)
        result["avg_order_value"] = result["avg_order_value"].round(2)
        result["avg_revenue_per_user"] = result["avg_revenue_per_user"].round(2)
        result["revenue_growth_pct"] = result["revenue_growth_pct"].round(2)
        result["rolling_revenue_sma_short"] = result["rolling_revenue_sma_short"].round(2)
        result["rolling_revenue_sma_long"] = result["rolling_revenue_sma_long"].round(2)
        result["cumulative_revenue"] = result["cumulative_revenue"].round(2)

        return result

    # =========================================================================
    # Churn Trend Computation
    # =========================================================================

    def compute_churn_trend(
        self,
        orders_df: pd.DataFrame,
        churn_labels_df: Optional[pd.DataFrame] = None,
        granularity: str = "daily",
        min_date: Optional[pd.Timestamp] = None,
        max_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Compute historical customer churn and retention trends for a specified granularity.

        Parameters:
            orders_df: Order-level DataFrame with customer IDs and timestamps.
            churn_labels_df: Optional pre-computed churn labels DataFrame.
            granularity: 'daily', 'weekly', or 'monthly'

        Returns:
            DataFrame adhering to Schema 6.5 (Churn Trend half).
        """
        if orders_df.empty:
            empty_df = pd.DataFrame(columns=CHURN_TREND_COLUMNS)
            empty_df["period_date"] = pd.to_datetime(empty_df["period_date"])
            empty_df["is_forecast"] = empty_df["is_forecast"].astype(bool)
            return empty_df

        df = orders_df.copy()
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
        df["period_date"] = self._normalize_timestamp_to_period(
            df["order_purchase_timestamp"], granularity
        )

        # 1. First order per customer (Acquisition period)
        customer_first = df.groupby("customer_unique_id")["order_purchase_timestamp"].min().reset_index()
        customer_first.rename(columns={"order_purchase_timestamp": "first_order_date"}, inplace=True)
        customer_first["acquisition_period"] = self._normalize_timestamp_to_period(
            customer_first["first_order_date"], granularity
        )

        # 2. Last order per customer
        customer_last = df.groupby("customer_unique_id")["order_purchase_timestamp"].max().reset_index()
        customer_last.rename(columns={"order_purchase_timestamp": "last_order_date"}, inplace=True)
        # Churn timestamp: when observation window elapsed after last purchase
        customer_last["churn_timestamp"] = customer_last["last_order_date"] + pd.Timedelta(days=self.return_window_days)
        customer_last["churn_period"] = self._normalize_timestamp_to_period(
            customer_last["churn_timestamp"], granularity
        )

        # 3. Merge customer acquisition and repeat order details
        df = df.merge(customer_first[["customer_unique_id", "first_order_date"]], on="customer_unique_id", how="left")
        df["is_repeat_order"] = df["order_purchase_timestamp"] > df["first_order_date"]

        # Aggregate period-level customer counts
        active_per_period = df.groupby("period_date")["customer_unique_id"].nunique().rename("active_customers")
        new_per_period = customer_first.groupby("acquisition_period")["customer_unique_id"].nunique().rename("new_customers")
        
        repeat_per_period = (
            df[df["is_repeat_order"]]
            .groupby("period_date")["customer_unique_id"]
            .nunique()
            .rename("repeat_customers")
        )

        # Churned customers whose return window elapsed in this period
        churned_per_period = customer_last.groupby("churn_period")["customer_unique_id"].nunique().rename("churned_customers")

        # Combine into time series with gap-filling
        freq = self._get_freq_alias(granularity)
        start_date = min_date or df["period_date"].min()
        end_date = max_date or df["period_date"].max()

        full_index = pd.date_range(start=start_date, end=end_date, freq=freq)
        trend_df = pd.DataFrame(index=full_index)

        trend_df = trend_df.join(active_per_period, how="left").fillna(0)
        trend_df = trend_df.join(new_per_period, how="left").fillna(0)
        trend_df = trend_df.join(repeat_per_period, how="left").fillna(0)
        trend_df = trend_df.join(churned_per_period, how="left").fillna(0)

        trend_df["active_customers"] = trend_df["active_customers"].astype(int)
        trend_df["new_customers"] = trend_df["new_customers"].astype(int)
        trend_df["repeat_customers"] = trend_df["repeat_customers"].astype(int)
        trend_df["churned_customers"] = trend_df["churned_customers"].astype(int)

        # Cumulative customer base observed up to period t
        cum_acquired = trend_df["new_customers"].cumsum()
        cum_churned = trend_df["churned_customers"].cumsum()
        trend_df["cumulative_churned"] = cum_churned

        # Active observed customer pool for churn rate denominator
        observed_base = cum_acquired.shift(1).fillna(trend_df["new_customers"])
        observed_base = observed_base.clip(lower=1)

        # Churn rate: churned customers in period / active observed base
        trend_df["churn_rate"] = np.where(
            observed_base > 0,
            (trend_df["churned_customers"] / observed_base).clip(0.0, 1.0),
            0.0,
        )
        trend_df["retention_rate"] = (1.0 - trend_df["churn_rate"]).clip(0.0, 1.0)

        # Rolling Churn Rate (smoothed)
        windows = self.rolling_windows.get(granularity.lower(), {"short": 7, "long": 30})
        short_w = windows.get("short", 7)
        trend_df["rolling_churn_rate"] = (
            trend_df["churn_rate"].rolling(window=short_w, min_periods=1).mean()
        )

        trend_df = trend_df.reset_index().rename(columns={"index": "period_date"})
        trend_df["granularity"] = granularity.lower()
        trend_df["is_forecast"] = False

        result = trend_df[CHURN_TREND_COLUMNS].copy()
        result["churn_rate"] = result["churn_rate"].round(4)
        result["retention_rate"] = result["retention_rate"].round(4)
        result["rolling_churn_rate"] = result["rolling_churn_rate"].round(4)

        return result

    # =========================================================================
    # Combined Master Trend Dataset (Schema 6.5)
    # =========================================================================

    def build_combined_trend(
        self,
        revenue_trend_df: pd.DataFrame,
        churn_trend_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge Revenue Trend and Churn Trend into the unified Schema 6.5 master table.
        """
        if revenue_trend_df.empty and churn_trend_df.empty:
            empty_df = pd.DataFrame(columns=COMBINED_TREND_COLUMNS)
            empty_df["period_date"] = pd.to_datetime(empty_df["period_date"])
            empty_df["is_forecast"] = empty_df["is_forecast"].astype(bool)
            return empty_df

        merged = pd.merge(
            revenue_trend_df,
            churn_trend_df[
                [
                    "granularity",
                    "period_date",
                    "active_customers",
                    "new_customers",
                    "repeat_customers",
                    "churned_customers",
                    "churn_rate",
                    "retention_rate",
                    "rolling_churn_rate",
                ]
            ],
            on=["granularity", "period_date"],
            how="outer",
        )

        merged["is_forecast"] = False
        merged = merged.sort_values(["granularity", "period_date"]).reset_index(drop=True)

        for col in COMBINED_TREND_COLUMNS:
            if col not in merged.columns:
                merged[col] = 0.0

        return merged[COMBINED_TREND_COLUMNS].copy()

    # =========================================================================
    # All-In-One Trend Pipeline Builder
    # =========================================================================

    def generate_all_trends(
        self,
        orders_df: pd.DataFrame,
        churn_labels_df: Optional[pd.DataFrame] = None,
        granularities: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Generate Daily, Weekly, and Monthly trends for both Revenue and Churn.
        """
        grans = granularities or list(self.SUPPORTED_GRANULARITIES)
        results: Dict[str, Dict[str, pd.DataFrame]] = {}

        all_rev_list = []
        all_churn_list = []
        all_master_list = []

        for gran in grans:
            rev_df = self.compute_revenue_trend(orders_df, granularity=gran)
            churn_df = self.compute_churn_trend(orders_df, churn_labels_df, granularity=gran)
            combined_df = self.build_combined_trend(rev_df, churn_df)

            validate_revenue_trend_schema(rev_df)
            validate_churn_trend_schema(churn_df)
            validate_combined_trend_schema(combined_df)

            results[gran] = {
                "revenue": rev_df,
                "churn": churn_df,
                "combined": combined_df,
            }

            all_rev_list.append(rev_df)
            all_churn_list.append(churn_df)
            all_master_list.append(combined_df)

        results["all_granularities_combined"] = {
            "revenue": pd.concat(all_rev_list, ignore_index=True) if all_rev_list else pd.DataFrame(),
            "churn": pd.concat(all_churn_list, ignore_index=True) if all_churn_list else pd.DataFrame(),
            "master": pd.concat(all_master_list, ignore_index=True) if all_master_list else pd.DataFrame(),
        }

        return results
