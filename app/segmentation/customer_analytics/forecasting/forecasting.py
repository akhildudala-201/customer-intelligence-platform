"""
Task 6 - Churn & Revenue Forecasting (Schema 6.5, forecast half)

Input : historical_trend_combined        (written by Person 5's Trends pipeline)
Output: forecast_revenue_trend,
        forecast_churn_trend,
        forecast_trend_combined          (MySQL, replaced on every run)
        + CSV copies in <project_root>/outputs/forecast_outputs/

Model : ETS (Holt-Winters) with a damped trend.
        Churn is modelled in logit space so forecasts always stay between 0 and 1.
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# This file lives at <root>/app/segmentation/customer_analytics/forecasting/,
# so parents[4] is the project root. Adding it to sys.path lets the script run
# from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from app.Database.database import engine  # noqa: E402  (shared engine, reads .env)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
GRANULARITY = "monthly"        # "daily" | "weekly" | "monthly"
FORECAST_HORIZON = 6           # periods to forecast ahead
TEST_HOLDOUT = 6               # recent periods held back to measure accuracy

INPUT_TABLE = "historical_trend_combined"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecast_outputs"

# Default pandas frequency and seasonal cycle length per granularity.
# The actual frequency is inferred from the data when possible (e.g. weekly
# periods may start on any weekday); the default is only a fallback.
FREQ_CONFIG = {
    "daily":   ("D", 7),
    "weekly":  ("W", 52),
    "monthly": ("MS", 12),
}


# ----------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------
def load_trend_data() -> pd.DataFrame:
    """Read historical (non-forecast) rows for one granularity, oldest first."""
    query = text(f"""
        SELECT *
        FROM {INPUT_TABLE}
        WHERE granularity = :granularity
          AND (is_forecast = 0 OR is_forecast IS NULL)
        ORDER BY period_date
    """)
    df = pd.read_sql(query, engine, params={"granularity": GRANULARITY})
    if df.empty:
        raise ValueError(f"No '{GRANULARITY}' rows found in {INPUT_TABLE}.")

    df["period_date"] = pd.to_datetime(df["period_date"])
    df = df.rename(columns={"total_revenue": "revenue"}).reset_index(drop=True)

    # The last few periods can be partial (data cut off mid-period), e.g. the
    # Olist export ends with Sep and Oct 2018 holding only a handful of orders.
    # Drop trailing periods while their order count is under 20% of the median
    # of the 6 periods before them; otherwise they look like a sudden collapse.
    while len(df) > 6:
        recent_median = df["order_count"].iloc[-7:-1].median()
        last_count = df["order_count"].iloc[-1]
        if not (recent_median > 0 and last_count < 0.2 * recent_median):
            break
        print(f"      Dropping incomplete period {df['period_date'].iloc[-1].date()} "
              f"({last_count} orders vs median {recent_median:.0f})")
        df = df.iloc[:-1]

    return df.set_index("period_date")


# ----------------------------------------------------------------------
# 2. Helpers
# ----------------------------------------------------------------------
def trim_leading_zeros(series: pd.Series, threshold: float = 1e-3) -> pd.Series:
    """Drop the early run of ~0 values (churn can't be measured until
    customers have enough tenure); otherwise the model mistakes the
    ramp-up for a trend and extrapolates it."""
    nonzero = np.flatnonzero(series.values > threshold)
    return series.iloc[nonzero[0]:] if len(nonzero) else series


def fill_zero_gaps(series: pd.Series) -> pd.Series:
    """Treat 0 values after the series has started as missing and interpolate.
    Exactly 0% churn across thousands of active customers (e.g. May-Jun 2017)
    is a measurement gap, not real; in logit space a 0 becomes about -9 against
    about -2 for normal months, which badly distorts the fitted trend."""
    return series.replace(0, np.nan).interpolate().bfill().ffill()


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def inv_logit(x):
    return 1 / (1 + np.exp(-x))


# ----------------------------------------------------------------------
# 3. Model
# ----------------------------------------------------------------------
def fit_and_forecast(series: pd.Series, horizon: int, freq: str, season_len: int):
    """Fit damped-trend ETS and forecast `horizon` periods with a 95% interval.

    Seasonality is only used when there are at least two full cycles of
    history; with less, ETS can't estimate it and the model runs trend-only.
    Returns (summary DataFrame with mean/pi_lower/pi_upper, model label).
    """
    series = series.astype(float).asfreq(freq)
    future_index = pd.date_range(series.index[-1], periods=horizon + 1, freq=freq)[1:]

    # A perfectly flat series (e.g. all zeros) has zero variance, so ETS's
    # log-likelihood takes log(0) and the fit is meaningless. The honest
    # forecast is simply "stays at that value".
    if series.nunique() <= 1:
        value = series.iloc[-1]
        flat = pd.DataFrame({"mean": value, "pi_lower": value, "pi_upper": value}, index=future_index)
        return flat, "constant (no variation in history)"

    seasonal = len(series) >= 2 * season_len

    fit = ETSModel(
        series,
        error="add",
        trend="add",
        damped_trend=True,  # stops the trend running away over the horizon
        seasonal="add" if seasonal else None,
        seasonal_periods=season_len if seasonal else None,
    ).fit(disp=False)

    summary = fit.get_prediction(start=len(series), end=len(series) + horizon - 1) \
                 .summary_frame(alpha=0.05)
    label = "ETS damped-trend" + (" + seasonal" if seasonal else "")

    # Record (rather than hide) a failed optimisation so it shows up in the output tables.
    if not fit.mle_retvals.get("converged", True):
        label += " [did not converge]"
    return summary, label


def evaluate_holdout(series, horizon, freq, season_len, inverse=None):
    """Train on all but the last `horizon` periods, score on those periods.
    `inverse` converts back to the original scale first (e.g. inv_logit for
    churn), so MAPE/RMSE are in business units rather than logit units."""
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    summary, _ = fit_and_forecast(train, horizon, freq, season_len)

    preds, actual = summary["mean"].values, test.values
    if inverse is not None:
        preds, actual = inverse(preds), inverse(actual)

    nonzero = np.abs(actual) > 1e-6  # skip ~0 actuals to avoid divide-by-zero in MAPE
    mape = np.mean(np.abs((actual[nonzero] - preds[nonzero]) / actual[nonzero])) * 100 if nonzero.any() else np.nan
    rmse = np.sqrt(np.mean((actual - preds) ** 2))
    return mape, rmse


# ----------------------------------------------------------------------
# 4. Pipeline
# ----------------------------------------------------------------------
def main():
    default_freq, season_len = FREQ_CONFIG[GRANULARITY]

    print(f"[1/4] Loading {GRANULARITY} history from {INPUT_TABLE}...")
    df = load_trend_data()
    freq = pd.infer_freq(df.index) or default_freq
    revenue = df["revenue"]
    churn_logit = fill_zero_gaps(trim_leading_zeros(df["churn_rate"])).apply(logit)
    print(f"      {len(df)} periods ({df.index.min().date()} to {df.index.max().date()}); "
          f"churn modelled from {churn_logit.index.min().date()}")

    print(f"[2/4] Holdout validation on last {TEST_HOLDOUT} periods...")
    rev_mape, rev_rmse = evaluate_holdout(revenue, TEST_HOLDOUT, freq, season_len)
    ch_mape, ch_rmse = evaluate_holdout(churn_logit, TEST_HOLDOUT, freq, season_len, inverse=inv_logit)
    print(f"      Revenue: MAPE {rev_mape:.2f}% | RMSE {rev_rmse:,.2f}")
    print(f"      Churn  : MAPE {ch_mape:.2f}% | RMSE {ch_rmse:.4f}")

    print(f"[3/4] Forecasting {FORECAST_HORIZON} periods ahead...")
    rev_fc, rev_model = fit_and_forecast(revenue, FORECAST_HORIZON, freq, season_len)
    rev_fc = rev_fc.clip(lower=0)  # revenue can't be negative; wide intervals can dip below 0
    churn_fc, churn_model = fit_and_forecast(churn_logit, FORECAST_HORIZON, freq, season_len)
    churn_fc = churn_fc.apply(inv_logit)  # back to a 0-1 rate

    # Shared columns for every output table
    base = {
        "period_date": rev_fc.index,
        "granularity": GRANULARITY,
        "generated_at": datetime.now().replace(microsecond=0),
    }

    revenue_out = pd.DataFrame({
        **base,
        "forecasted_revenue": rev_fc["mean"].round(2).values,
        "ci_lower_revenue": rev_fc["pi_lower"].round(2).values,
        "ci_upper_revenue": rev_fc["pi_upper"].round(2).values,
        "holdout_mape_pct": round(rev_mape, 2),
        "model_used": rev_model,
    })

    churn_out = pd.DataFrame({
        **base,
        "forecasted_churn_rate": churn_fc["mean"].round(4).values,
        "ci_lower_churn": churn_fc["pi_lower"].round(4).values,
        "ci_upper_churn": churn_fc["pi_upper"].round(4).values,
        "holdout_mape_pct": round(ch_mape, 2),
        "model_used": f"{churn_model} (logit)",
    })

    combined_out = revenue_out.merge(
        churn_out, on=["period_date", "granularity", "generated_at"], suffixes=("_revenue", "_churn")
    )

    print("[4/4] Saving results...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "forecast_revenue_trend": revenue_out,
        "forecast_churn_trend": churn_out,
        "forecast_trend_combined": combined_out,
    }
    for table, frame in outputs.items():
        frame.to_sql(table, engine, if_exists="replace", index=False)
        frame.to_csv(OUTPUT_DIR / f"{table}.csv", index=False)
        print(f"      Saved {len(frame)} rows -> MySQL `{table}` + {table}.csv")

    print("\n" + "=" * 70)
    print("FORECASTING COMPLETE")
    print(combined_out[["period_date", "forecasted_revenue", "forecasted_churn_rate"]].to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()