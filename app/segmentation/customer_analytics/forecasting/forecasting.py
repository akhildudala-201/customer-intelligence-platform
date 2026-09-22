

from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import text
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from app.Database.database import engine

np.random.seed(42)

GRANULARITY = "monthly"                        # "daily", "weekly", or "monthly"
INPUT_TABLE = "historical_trend_combined"
FORECAST_HORIZON = 6                           # periods ahead to forecast
SEASONAL_PERIOD = 12                           # 12=monthly/yearly, 52=weekly, 7=daily-weekly
TEST_HOLDOUT = 6                               # periods held out to validate accuracy
OUTPUT_DIR = Path(__file__).resolve().parents[4] / "outputs" / "forecast_outputs"   # app/outputs/forecast_outputs


def get_engine():
    return engine


# ----------------------------------------------------------------------
# STEP 1 - LOAD trend data from MySQL
# Real historical_trend_combined schema:
#   granularity, period_date, total_revenue, order_count, unique_customers,
#   avg_order_value, avg_revenue_per_user, revenue_growth_pct,
#   active_customers, new_customers, repeat_customers, churned_customers,
#   churn_rate, retention_rate, rolling_churn_rate, is_forecast
# ----------------------------------------------------------------------
def load_trend_data(engine):
    query = text(f"""
        SELECT *
        FROM {INPUT_TABLE}
        WHERE granularity = :granularity
          AND (is_forecast = 0 OR is_forecast IS NULL)
        ORDER BY period_date ASC
    """)
    df = pd.read_sql(query, engine, params={"granularity": GRANULARITY})

    if df.empty:
        raise ValueError(
            f"No rows returned from `{INPUT_TABLE}` for granularity='{GRANULARITY}'. "
            f"Check the table/column names match your actual schema."
        )

    df = df.rename(columns={"period_date": "period_start", "total_revenue": "revenue"})
    df["period_start"] = pd.to_datetime(df["period_start"])
    df = df.sort_values("period_start").reset_index(drop=True)

    # Guard against a partial/in-progress trailing period (e.g. data pulled
    # mid-month, so the most recent row only has a few days of orders).
    # If the last period's order_count is far below the recent median, it's
    # almost certainly incomplete rather than a real collapse -> drop it.
    if len(df) > 6 and "order_count" in df.columns:
        recent_median = df["order_count"].iloc[-7:-1].median()
        last_count = df["order_count"].iloc[-1]
        if recent_median > 0 and last_count < 0.2 * recent_median:
            print(
                f"      [load_trend_data] Dropping likely-incomplete trailing period "
                f"{df['period_start'].iloc[-1].date()} (order_count={last_count} vs "
                f"recent median={recent_median:.0f})"
            )
            df = df.iloc[:-1].reset_index(drop=True)

    return df


# ----------------------------------------------------------------------
# STEP 2 - TRANSFORMS (churn is bounded 0-1, model it in logit space)
# ----------------------------------------------------------------------
def trim_leading_zeros(raw_series, threshold=1e-3):
    """
    Drop leading periods where the raw value is ~0. For churn specifically,
    early zero periods usually mean "not enough customer tenure yet to
    measure churn" rather than genuine zero churn -- treating that as real
    signal makes the model see a steep 0 -> ~0.08 climb and extrapolate
    that climb forever. Trimming lets the model fit only the mature,
    steady-state period.
    """
    nonzero_idx = np.where(raw_series.values > threshold)[0]
    if len(nonzero_idx) == 0:
        return raw_series
    return raw_series.iloc[nonzero_idx[0]:]


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def inv_logit(x):
    return 1 / (1 + np.exp(-x))


# ----------------------------------------------------------------------
# STEP 3 - FIT + FORECAST a single series with ETS (Holt-Winters family)
# Falls back to a non-seasonal model automatically if there isn't enough
# history (< 2 full seasonal cycles) to estimate seasonality reliably.
# ----------------------------------------------------------------------
def fit_and_forecast(series, horizon, seasonal_periods, seasonal="add", trend="add", freq="MS"):
    series = series.astype(float)
    series.index = pd.DatetimeIndex(series.index, freq=freq)

    use_seasonal = seasonal_periods is not None and len(series) >= 2 * seasonal_periods
    effective_seasonal = seasonal if use_seasonal else None
    effective_seasonal_periods = seasonal_periods if use_seasonal else None


    model = ETSModel(
        series,
        error="add",
        trend=trend,
        damped_trend=(trend is not None),
        seasonal=effective_seasonal,
        seasonal_periods=effective_seasonal_periods,
    )
    fit = model.fit(disp=False)
    pred = fit.get_prediction(start=len(series), end=len(series) + horizon - 1)
    summary = pred.summary_frame(alpha=0.05)  # 95% CI
    return fit, summary


# ----------------------------------------------------------------------
# STEP 4 - VALIDATE ON HOLDOUT (time-based split, not random)
# ----------------------------------------------------------------------
def evaluate_holdout(series, horizon, **kwargs):
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    _, summary = fit_and_forecast(train, horizon, **kwargs)
    preds, actual = summary["mean"].values, test.values

    # Guard against divide-by-zero in MAPE for near-zero actual periods
    nonzero_mask = np.abs(actual) > 1e-6
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((actual[nonzero_mask] - preds[nonzero_mask]) / actual[nonzero_mask])) * 100
    else:
        mape = np.nan
    rmse = np.sqrt(np.mean((actual - preds) ** 2))
    return mape, rmse


# ----------------------------------------------------------------------
# STEP 5 - WRITE forecast tables back to MySQL
# ----------------------------------------------------------------------
def write_forecast_to_mysql(engine, revenue_df, churn_df, combined_df):
    revenue_df.to_sql("forecast_revenue_trend", engine, if_exists="replace", index=False)
    churn_df.to_sql("forecast_churn_trend", engine, if_exists="replace", index=False)
    combined_df.to_sql("forecast_trend_combined", engine, if_exists="replace", index=False)
    print("[MySQL] Saved forecast_revenue_trend, forecast_churn_trend, forecast_trend_combined")


# ----------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------
def main():
    freq_map = {"daily": "D", "weekly": "W", "monthly": "MS"}
    offset_map = {"daily": pd.DateOffset(days=1), "weekly": pd.DateOffset(weeks=1), "monthly": pd.DateOffset(months=1)}
    freq = freq_map[GRANULARITY]

    print(f"[1/4] Loading trend data from MySQL (table: {INPUT_TABLE}, granularity: {GRANULARITY})...")
    df = load_trend_data(engine)
    df = df.set_index("period_start")
    print(f"      Loaded {len(df)} clean periods "
          f"({df.index.min().date()} to {df.index.max().date()})")

    revenue_series = df["revenue"]
    churn_series = trim_leading_zeros(df["churn_rate"])
    churn_logit_series = churn_series.apply(logit)
    print(f"      Churn series trimmed to {len(churn_series)} mature periods "
          f"(from {churn_series.index.min().date()} onward)")

    print(f"[2/4] Validating on last {TEST_HOLDOUT} periods...")
    rev_mape, rev_rmse = evaluate_holdout(revenue_series, TEST_HOLDOUT, seasonal_periods=SEASONAL_PERIOD, freq=freq)
    churn_mape, churn_rmse = evaluate_holdout(churn_logit_series, TEST_HOLDOUT, seasonal_periods=SEASONAL_PERIOD, freq=freq)
    print(f"      Revenue -> MAPE: {rev_mape:.2f}%  RMSE: {rev_rmse:,.2f}")
    print(f"      Churn (logit space) -> MAPE: {churn_mape:.2f}%  RMSE: {churn_rmse:.4f}")

    print(f"[3/4] Refitting on full history and forecasting {FORECAST_HORIZON} periods forward...")
    _, revenue_fc = fit_and_forecast(revenue_series, FORECAST_HORIZON, SEASONAL_PERIOD, freq=freq)
    _, churn_fc_logit = fit_and_forecast(churn_logit_series, FORECAST_HORIZON, SEASONAL_PERIOD, freq=freq)

    churn_mean = inv_logit(churn_fc_logit["mean"])
    churn_lo = inv_logit(churn_fc_logit["pi_lower"])
    churn_hi = inv_logit(churn_fc_logit["pi_upper"])

    future_dates = pd.date_range(start=df.index[-1] + offset_map[GRANULARITY], periods=FORECAST_HORIZON, freq=freq)

    revenue_out = pd.DataFrame({
        "period_start": future_dates,
        "granularity": GRANULARITY,
        "forecasted_revenue": revenue_fc["mean"].round(2).values,
        "ci_lower_revenue": revenue_fc["pi_lower"].round(2).values,
        "ci_upper_revenue": revenue_fc["pi_upper"].round(2).values,
        "model_used": "ETS Holt-Winters (additive)",
    })

    churn_out = pd.DataFrame({
        "period_start": future_dates,
        "granularity": GRANULARITY,
        "forecasted_churn_rate": churn_mean.round(4).values,
        "ci_lower_churn": churn_lo.round(4).values,
        "ci_upper_churn": churn_hi.round(4).values,
        "model_used": "ETS Holt-Winters (logit-transformed)",
    })

    combined_out = revenue_out.merge(churn_out, on=["period_start", "granularity"], suffixes=("_rev", "_churn"))

    print("[4/4] Writing forecast tables back to MySQL...")
    write_forecast_to_mysql(engine, revenue_out, churn_out, combined_out)

    # Local CSV backups as well
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    revenue_out.to_csv(OUTPUT_DIR / "forecast_revenue_trend.csv", index=False)
    churn_out.to_csv(OUTPUT_DIR / "forecast_churn_trend.csv", index=False)
    combined_out.to_csv(OUTPUT_DIR / "forecast_trend_combined.csv", index=False)
    print(f"      Local CSV backups written to: {OUTPUT_DIR.resolve()}")

    print("\n" + "=" * 70)
    print("FORECASTING COMPLETE")
    print(combined_out.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()