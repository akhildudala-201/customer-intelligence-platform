"""
app/segmentation/customer_analytics/Trends/trend_engine.py

Core Historical Trend Analysis Engine (Person 5).
Produces aggregated, continuous historical time-series trends for Churn and Revenue
at Daily, Weekly, and Monthly levels, conforming to Schema 6.5 (trend half).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    from app.segmentation.customer_analytics.Trends.schemas import (
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
        reference_date: Optional[Union[str, pd.Timestamp]] = "2018-10-17",
        rolling_windows: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        self.return_window_days = int(return_window_days)
        self.reference_date = pd.Timestamp(reference_date) if reference_date else pd.Timestamp("2018-10-17")
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
        # First period has NA/None growth
        if len(resampled) > 0:
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

        # Float precision rounding
        result["total_revenue"] = result["total_revenue"].round(2)
        result["avg_order_value"] = result["avg_order_value"].round(2)
        result["avg_revenue_per_user"] = result["avg_revenue_per_user"].round(2)
        result["revenue_growth_pct"] = result["revenue_growth_pct"].round(2)
        result["rolling_revenue_sma_short"] = result["rolling_revenue_sma_short"].round(2)
        result["rolling_revenue_sma_long"] = result["rolling_revenue_sma_long"].round(2)
        result["cumulative_revenue"] = result["cumulative_revenue"].round(2)

        return result

    # =========================================================================
    # Churn Trend Computation (Issue 1, 2, 3 Fixed)
    # =========================================================================

    def compute_churn_trend(
        self,
        orders_df: pd.DataFrame,
        churn_labels_df: Optional[pd.DataFrame] = None,
        granularity: str = "daily",
        reference_date: Optional[pd.Timestamp] = None,
        min_date: Optional[pd.Timestamp] = None,
        max_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Compute historical customer churn and retention trends for a specified granularity.

        Parameters:
            orders_df: Order-level DataFrame with customer IDs and timestamps.
            churn_labels_df: Pre-computed churn labels DataFrame containing
                             ['customer_unique_id', 'first_order_date', 'last_order_date',
                              'label', 'censored']. If None, computed from orders.
            granularity: 'daily', 'weekly', or 'monthly'
            reference_date: Timestamp cutoff for churn observation.

        Returns:
            DataFrame adhering to Schema 6.5 (Churn Trend half).
        """
        if orders_df.empty:
            empty_df = pd.DataFrame(columns=CHURN_TREND_COLUMNS)
            empty_df["period_date"] = pd.to_datetime(empty_df["period_date"])
            empty_df["is_censored"] = empty_df["is_censored"].astype(bool)
            empty_df["is_forecast"] = empty_df["is_forecast"].astype(bool)
            return empty_df

        ref_date = reference_date or self.reference_date

        df_orders = orders_df.copy()
        df_orders["order_purchase_timestamp"] = pd.to_datetime(df_orders["order_purchase_timestamp"])
        df_orders["period_date"] = self._normalize_timestamp_to_period(
            df_orders["order_purchase_timestamp"], granularity
        )

        # 1. Obtain or build customer churn labels dataset
        if churn_labels_df is None or churn_labels_df.empty:
            # Dynamically compute customer-level first/last orders and labels
            df_orders_sorted = df_orders.sort_values(
                ["customer_unique_id", "order_purchase_timestamp"]
            ).reset_index(drop=True)
            grouped_cust = df_orders_sorted.groupby("customer_unique_id")

            labels = grouped_cust.agg(
                first_order_date=("order_purchase_timestamp", "min"),
                last_order_date=("order_purchase_timestamp", "max"),
                num_valid_orders=("order_id", "nunique"),
            ).reset_index()

            labels["days_since_last_order"] = (
                ref_date - labels["last_order_date"]
            ).dt.total_seconds() / 86400.0

            df_orders_sorted["prev_order_date"] = df_orders_sorted.groupby(
                "customer_unique_id"
            )["order_purchase_timestamp"].shift(1)
            df_orders_sorted["gap_to_prev"] = (
                df_orders_sorted["order_purchase_timestamp"]
                - df_orders_sorted["prev_order_date"]
            ).dt.total_seconds() / 86400.0

            repeat_customers = set(
                df_orders_sorted.loc[
                    df_orders_sorted["gap_to_prev"] <= self.return_window_days,
                    "customer_unique_id",
                ]
            )
            labels["has_repeat"] = labels["customer_unique_id"].isin(repeat_customers)
            labels["censored"] = (
                (~labels["has_repeat"])
                & (labels["days_since_last_order"] < self.return_window_days)
            )

            labels["label"] = pd.NA
            labels.loc[labels["has_repeat"], "label"] = 0
            labels.loc[
                (~labels["has_repeat"])
                & (labels["days_since_last_order"] >= self.return_window_days),
                "label",
            ] = 1
            labels = labels.drop(columns=["has_repeat"])
            labels["label"] = labels["label"].astype("Int64")
        else:
            labels = churn_labels_df.copy()
            labels["first_order_date"] = pd.to_datetime(labels["first_order_date"])
            labels["last_order_date"] = pd.to_datetime(labels["last_order_date"])

        # 2. Acquisition Periods (New Customers)
        labels["acquisition_period"] = self._normalize_timestamp_to_period(
            labels["first_order_date"], granularity
        )
        new_per_period = (
            labels.groupby("acquisition_period")["customer_unique_id"]
            .nunique()
            .rename("new_customers")
        )

        # 3. Repeat Orders per Period
        df_orders = df_orders.merge(
            labels[["customer_unique_id", "first_order_date"]],
            on="customer_unique_id",
            how="left",
        )
        df_orders["is_repeat_order"] = (
            df_orders["order_purchase_timestamp"] > df_orders["first_order_date"]
        )
        repeat_per_period = (
            df_orders[df_orders["is_repeat_order"]]
            .groupby("period_date")["customer_unique_id"]
            .nunique()
            .rename("repeat_customers")
        )

        # 4. Churn Events from Confirmed Churned Customers (Issue 1 Fix)
        # Churn event date is when return_window expired after their last order
        churned_cust = labels[labels["label"] == 1].copy()
        if not churned_cust.empty:
            churned_cust["churn_date"] = churned_cust["last_order_date"] + pd.Timedelta(
                days=self.return_window_days
            )
            churned_cust["churn_period"] = self._normalize_timestamp_to_period(
                churned_cust["churn_date"], granularity
            )
            churned_per_period = (
                churned_cust.groupby("churn_period")["customer_unique_id"]
                .nunique()
                .rename("churned_customers")
            )
        else:
            churned_per_period = pd.Series(dtype=int, name="churned_customers")

        # 5. Continuous Timeline Index (Gap-Filling up to max date / reference date)
        freq = self._get_freq_alias(granularity)
        start_date = min_date or min(df_orders["period_date"].min(), labels["acquisition_period"].min())

        max_candidates = [df_orders["period_date"].max()]
        if not churned_cust.empty:
            max_candidates.append(churned_cust["churn_period"].max())
        if ref_date is not None:
            ref_period = self._normalize_timestamp_to_period(pd.Series([ref_date]), granularity).iloc[0]
            max_candidates.append(ref_period)

        end_date = max_date or max(max_candidates)

        full_index = pd.date_range(start=start_date, end=end_date, freq=freq)
        trend_df = pd.DataFrame(index=full_index)

        trend_df = trend_df.join(new_per_period, how="left").fillna(0)
        trend_df = trend_df.join(repeat_per_period, how="left").fillna(0)
        trend_df = trend_df.join(churned_per_period, how="left").fillna(0)

        trend_df["new_customers"] = trend_df["new_customers"].astype(int)
        trend_df["repeat_customers"] = trend_df["repeat_customers"].astype(int)
        trend_df["churned_customers"] = trend_df["churned_customers"].astype(int)

        # 6. Cumulative Metrics & Survivorship Active Base (Issue 2 & 3 Fix)
        trend_df["cumulative_acquired"] = trend_df["new_customers"].cumsum()
        trend_df["cumulative_churned"] = trend_df["churned_customers"].cumsum()

        # Surviving customer base entering period t:
        # cumulative acquired up to t-1 minus cumulative churned up to t-1, plus new acquisitions in t
        surviving_prior = (
            trend_df["cumulative_acquired"].shift(1).fillna(0)
            - trend_df["cumulative_churned"].shift(1).fillna(0)
        )
        # Active customer base entering/during period t:
        surviving_base = surviving_prior + trend_df["new_customers"]
        trend_df["active_customers"] = surviving_base.clip(lower=0).astype(int)

        # Churn Rate using survivorship active base denominator
        # For periods with zero active customers, churn rate is 0.0
        trend_df["churn_rate"] = np.where(
            trend_df["active_customers"] > 0,
            (trend_df["churned_customers"] / trend_df["active_customers"]).clip(0.0, 1.0),
            0.0,
        )
        trend_df["retention_rate"] = (1.0 - trend_df["churn_rate"]).clip(0.0, 1.0)

        # Rolling Churn Rate (smoothed moving average)
        windows = self.rolling_windows.get(granularity.lower(), {"short": 7, "long": 30})
        short_w = windows.get("short", 7)
        trend_df["rolling_churn_rate"] = (
            trend_df["churn_rate"].rolling(window=short_w, min_periods=1).mean()
        )

        trend_df = trend_df.reset_index().rename(columns={"index": "period_date"})
        trend_df["granularity"] = granularity.lower()

        # 7. Right-Censoring Flag (Issue 1 Fix)
        # Periods whose return window extends beyond reference_date have incomplete churn observation
        if ref_date is not None:
            censoring_threshold_date = ref_date - pd.Timedelta(days=self.return_window_days)
            trend_df["is_censored"] = trend_df["period_date"] > censoring_threshold_date
        else:
            trend_df["is_censored"] = False

        trend_df["is_forecast"] = False

        # Format and validate Schema 6.5 contract
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
            empty_df["is_censored"] = empty_df["is_censored"].astype(bool)
            empty_df["is_forecast"] = empty_df["is_forecast"].astype(bool)
            return empty_df

        if revenue_trend_df.empty:
            merged = churn_trend_df.copy()
            for col in COMBINED_TREND_COLUMNS:
                if col not in merged.columns:
                    merged[col] = 0.0
            return merged[COMBINED_TREND_COLUMNS].copy()

        if churn_trend_df.empty:
            merged = revenue_trend_df.copy()
            for col in COMBINED_TREND_COLUMNS:
                if col not in merged.columns:
                    merged[col] = False if col in ("is_censored", "is_forecast") else 0.0
            return merged[COMBINED_TREND_COLUMNS].copy()

        churn_cols = [
            "granularity",
            "period_date",
            "active_customers",
            "new_customers",
            "repeat_customers",
            "churned_customers",
            "churn_rate",
            "retention_rate",
            "rolling_churn_rate",
            "is_censored",
        ]

        merged = pd.merge(
            revenue_trend_df,
            churn_trend_df[[c for c in churn_cols if c in churn_trend_df.columns]],
            on=["granularity", "period_date"],
            how="outer",
        )

        merged["is_forecast"] = False
        if "is_censored" not in merged.columns:
            merged["is_censored"] = False
        else:
            merged["is_censored"] = merged["is_censored"].fillna(False).astype(bool)

        merged = merged.sort_values(["granularity", "period_date"]).reset_index(drop=True)

        for col in COMBINED_TREND_COLUMNS:
            if col not in merged.columns:
                merged[col] = 0.0
            else:
                if col in ("is_censored", "is_forecast"):
                    merged[col] = merged[col].fillna(False).astype(bool)
                elif col in ("granularity", "period_date"):
                    pass
                elif col == "cumulative_revenue":
                    merged[col] = merged[col].ffill().fillna(0.0)
                else:
                    merged[col] = merged[col].fillna(0.0)

        return merged[COMBINED_TREND_COLUMNS].copy()

    # =========================================================================
    # All-In-One Trend Pipeline Builder (Issue 4 Validation Enforcement)
    # =========================================================================

    def generate_all_trends(
        self,
        orders_df: pd.DataFrame,
        churn_labels_df: Optional[pd.DataFrame] = None,
        granularities: Optional[List[str]] = None,
        reference_date: Optional[pd.Timestamp] = None,
        raise_on_validation_error: bool = True,
    ) -> Dict[str, Dict[str, Union[pd.DataFrame, dict]]]:
        """
        Generate Daily, Weekly, and Monthly trends for both Revenue and Churn.
        Enforces Schema 6.5 validation and surfaces errors.
        """
        ref_date = reference_date or self.reference_date
        grans = granularities or list(self.SUPPORTED_GRANULARITIES)
        results: Dict[str, Dict[str, Union[pd.DataFrame, dict]]] = {}

        all_rev_list = []
        all_churn_list = []
        all_master_list = []
        validation_reports = {}

        for gran in grans:
            rev_df = self.compute_revenue_trend(orders_df, granularity=gran)
            churn_df = self.compute_churn_trend(
                orders_df,
                churn_labels_df=churn_labels_df,
                granularity=gran,
                reference_date=ref_date,
            )
            combined_df = self.build_combined_trend(rev_df, churn_df)

            # Strict Schema 6.5 Validations (Issue 4 Fix)
            val_rev = validate_revenue_trend_schema(rev_df)
            val_churn = validate_churn_trend_schema(churn_df)
            val_comb = validate_combined_trend_schema(combined_df)

            gran_errors = []
            if not val_rev.is_valid:
                gran_errors.append(f"Revenue Trend ({gran}): {val_rev.errors}")
            if not val_churn.is_valid:
                gran_errors.append(f"Churn Trend ({gran}): {val_churn.errors}")
            if not val_comb.is_valid:
                gran_errors.append(f"Combined Trend ({gran}): {val_comb.errors}")

            if gran_errors and raise_on_validation_error:
                raise ValueError(
                    f"Schema 6.5 validation failure on granularity '{gran}':\n"
                    + "\n".join(gran_errors)
                )

            validation_reports[gran] = {
                "revenue": {
                    "is_valid": val_rev.is_valid,
                    "errors": val_rev.errors,
                    "row_count": val_rev.row_count,
                },
                "churn": {
                    "is_valid": val_churn.is_valid,
                    "errors": val_churn.errors,
                    "row_count": val_churn.row_count,
                },
                "combined": {
                    "is_valid": val_comb.is_valid,
                    "errors": val_comb.errors,
                    "row_count": val_comb.row_count,
                },
            }

            results[gran] = {
                "revenue": rev_df,
                "churn": churn_df,
                "combined": combined_df,
                "validation": validation_reports[gran],
            }

            all_rev_list.append(rev_df)
            all_churn_list.append(churn_df)
            all_master_list.append(combined_df)

        master_rev = pd.concat(all_rev_list, ignore_index=True) if all_rev_list else pd.DataFrame()
        master_churn = pd.concat(all_churn_list, ignore_index=True) if all_churn_list else pd.DataFrame()
        master_comb = pd.concat(all_master_list, ignore_index=True) if all_master_list else pd.DataFrame()

        results["all_granularities_combined"] = {
            "revenue": master_rev,
            "churn": master_churn,
            "master": master_comb,
            "validation": validation_reports,
        }

        return results
