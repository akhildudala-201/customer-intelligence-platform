
import os
import threading
import time
from datetime import date
from typing import Callable, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

CHURN_TABLE = "historical_churn_trend"
REVENUE_TABLE = "historical_revenue_trend"
COMBINED_TABLE = "historical_trend_combined"
COHORT_TABLE = "customer_cohort_analysis"

# Person 4's column names contain spaces/brackets; the API exposes clean names.
COHORT_RENAME = {
    "retention_rate (%)": "retention_rate_pct",
    "repeat_purchase_rate (%)": "repeat_purchase_rate_pct",
    "cumulative_repeat_purchase_rate (%)": "cumulative_repeat_purchase_rate_pct",
    "final_churn_rate (%)": "final_churn_rate_pct",
    "monthly_churn_rate (%)": "monthly_churn_rate_pct",
    "cumulative_churn_rate (%)": "cumulative_churn_rate_pct",
}

# Seconds to keep a table in memory. The tables only change when the pipeline
# re-runs, so a short cache saves a DB round-trip on every dashboard request.
# Set ANALYTICS_CACHE_TTL_SECONDS=0 to always read fresh.
CACHE_TTL = float(os.getenv("ANALYTICS_CACHE_TTL_SECONDS", "300"))

_cache: dict = {}
_cache_lock = threading.Lock()


def clear_cache() -> None:
    """Drop all cached tables (e.g. right after a pipeline run, or in tests)."""
    with _cache_lock:
        _cache.clear()


def _cached(key: tuple, loader: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Return a cached DataFrame, reloading it once it is older than CACHE_TTL.
    Callers must not modify the returned frame (it is shared)."""
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL:
            return hit[1]
    frame = loader()
    if CACHE_TTL > 0:
        with _cache_lock:
            _cache[key] = (now, frame)
    return frame


def _read_sql(engine: Engine, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
    """Run a query; a missing table or unreachable DB becomes a clear 503."""
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)
    except (SQLAlchemyError, pd.errors.DatabaseError) as exc:  # pandas re-wraps DB errors
        raise HTTPException(
            status_code=503,
            detail="Analytics data is not available. Make sure the database is "
                   "running and the pipeline has been run.",
        ) from exc


def to_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe dicts: NaN/NaT become None (JSON has no NaN)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


# ----------------------------------------------------------------------
# Trends
# ----------------------------------------------------------------------
def _load_trend(engine: Engine, table: str, granularity: str) -> pd.DataFrame:
    df = _read_sql(
        engine,
        f"SELECT * FROM {table} WHERE granularity = :granularity",  # table name is a constant, never user input
        {"granularity": granularity},
    )
    df["period_date"] = pd.to_datetime(df["period_date"]).dt.date
    if "is_forecast" in df:                        # history only
        df = df[~df["is_forecast"].fillna(False).astype(bool)]
    if "is_censored" in df:                        # MySQL returns tinyint 0/1
        df["is_censored"] = df["is_censored"].astype("boolean")
    return df.sort_values("period_date").reset_index(drop=True)


def get_trend(
    engine: Engine,
    table: str,
    granularity: str,
    columns: list[str],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """One granularity of a trend table, optional date range, fixed column set.
    Columns missing from the table (e.g. an older run without is_censored)
    come back as None instead of failing."""
    df = _cached((id(engine), table, granularity), lambda: _load_trend(engine, table, granularity))
    mask = pd.Series(True, index=df.index)
    if start_date:
        mask &= df["period_date"] >= start_date
    if end_date:
        mask &= df["period_date"] <= end_date
    return df.loc[mask].reindex(columns=columns)


def get_new_vs_repeat(engine: Engine, granularity: str,
                      start_date: Optional[date] = None, end_date: Optional[date] = None) -> pd.DataFrame:
    df = get_trend(engine, CHURN_TABLE, granularity,
                   ["period_date", "new_customers", "repeat_customers"], start_date, end_date)
    total = df["new_customers"] + df["repeat_customers"]
    # Share of repeat customers among everyone who bought; None when nobody bought
    df["repeat_share_pct"] = (df["repeat_customers"] / total.where(total > 0) * 100).round(2)
    return df


# ----------------------------------------------------------------------
# Cohorts
# ----------------------------------------------------------------------
def _load_cohorts(engine: Engine) -> pd.DataFrame:
    df = _read_sql(engine, f"SELECT * FROM {COHORT_TABLE}").rename(columns=COHORT_RENAME)
    # "Mar 2017" -> 2017-03-01 for correct date ordering, and "2017-03" for URLs
    cohort_start = pd.to_datetime(df["cohort"], format="%b %Y")
    df["label"] = df["cohort"]
    df["cohort"] = cohort_start.dt.strftime("%Y-%m")
    df["_cohort_start"] = cohort_start
    # "M10" -> 10, so months sort numerically (text sort would put M10 before M2)
    df["_month_no"] = df["relative_month"].str[1:].astype(int)
    df = df.rename(columns={"relative_month": "month"})
    if "generated_date" in df:
        df["generated_date"] = pd.to_datetime(df["generated_date"]).dt.date
    return df.sort_values(["_cohort_start", "_month_no"]).reset_index(drop=True)


def get_cohorts(engine: Engine, cohort_from: Optional[str] = None,
                cohort_to: Optional[str] = None) -> pd.DataFrame:
    """All cohort rows (cohort x month), optionally limited to a cohort range.
    'YYYY-MM' strings compare correctly as text."""
    df = _cached((id(engine), COHORT_TABLE), lambda: _load_cohorts(engine))
    mask = pd.Series(True, index=df.index)
    if cohort_from:
        mask &= df["cohort"] >= cohort_from
    if cohort_to:
        mask &= df["cohort"] <= cohort_to
    return df.loc[mask]


def generated_date(df: pd.DataFrame) -> Optional[date]:
    return df["generated_date"].max() if "generated_date" in df and not df.empty else None


def retention_matrix(df: pd.DataFrame) -> dict:
    """Cohort x month grid of retention %. Months a cohort hasn't reached yet
    stay None (not 0), so recent cohorts aren't shown as losing everyone."""
    if df.empty:
        return {"cohorts": [], "labels": [], "months": [], "values": []}
    # "YYYY-MM" keys sort in date order, so the pivot's sorted index is correct
    grid = (df.pivot(index="cohort", columns="_month_no", values="retention_rate_pct")
              .reindex(columns=range(int(df["_month_no"].max()) + 1)))
    labels = df.drop_duplicates("cohort").set_index("cohort")["label"]
    return {
        "cohorts": grid.index.tolist(),
        "labels": labels.reindex(grid.index).tolist(),
        "months": [f"M{m}" for m in grid.columns],
        "values": [[None if pd.isna(v) else float(v) for v in row] for row in grid.to_numpy()],
    }


def cohort_series(df: pd.DataFrame, month_fields: list[str],
                  extra: Callable[[pd.DataFrame], dict]) -> list[dict]:
    """One entry per cohort: its month-by-month values for `month_fields`,
    plus cohort-level fields computed by `extra(cohort_rows)`."""
    out = []
    for (cohort, label), rows in df.groupby(["cohort", "label"], sort=False):
        out.append({
            "cohort": cohort,
            "label": label,
            **extra(rows),
            "months": to_records(rows[["month", *month_fields]]),
        })
    return out


def last_value(rows: pd.DataFrame, column: str):
    """Value in the cohort's latest month (None if missing)."""
    if column not in rows:
        return None
    value = rows[column].iloc[-1]
    return None if pd.isna(value) else value


def first_value(rows: pd.DataFrame, column: str):
    """Cohort-level value repeated on every row (e.g. final churn, size)."""
    if column not in rows:
        return None
    values = rows[column].dropna()
    return values.iloc[0] if len(values) else None


def _at_month(rows: pd.DataFrame, column: str, month_no: int):
    value = rows.loc[rows["_month_no"] == month_no, column]
    return None if value.empty or pd.isna(value.iloc[0]) else float(value.iloc[0])


def cohort_summary(df: pd.DataFrame) -> list[dict]:
    rows_out = []
    for (cohort, label), rows in df.groupby(["cohort", "label"], sort=False):
        size = first_value(rows, "cohort_size")
        rows_out.append({
            "cohort": cohort,
            "label": label,
            "cohort_size": None if size is None else int(size),
            "months_observed": len(rows),
            "m1_retention_pct": _at_month(rows, "retention_rate_pct", 1),
            "m3_retention_pct": _at_month(rows, "retention_rate_pct", 3),
            "final_churn_rate_pct": first_value(rows, "final_churn_rate_pct"),
            "cumulative_repeat_purchase_rate_pct": last_value(rows, "cumulative_repeat_purchase_rate_pct"),
            "cumulative_average_revenue": last_value(rows, "cumulative_average_revenue"),
        })
    return rows_out


DETAIL_FIELDS = [
    "retention_rate_pct", "repeat_purchase_rate_pct", "cumulative_repeat_purchase_rate_pct",
    "monthly_churn_rate_pct", "cumulative_churn_rate_pct", "cumulative_average_revenue",
]


def cohort_detail(df: pd.DataFrame, cohort: str) -> Optional[dict]:
    rows = df[df["cohort"] == cohort]
    if rows.empty:
        return None
    size = first_value(rows, "cohort_size")
    return {
        "cohort": cohort,
        "label": rows["label"].iloc[0],
        "cohort_size": None if size is None else int(size),
        "final_churn_rate_pct": first_value(rows, "final_churn_rate_pct"),
        "generated_date": generated_date(rows),
        "months": to_records(rows.reindex(columns=["month", *DETAIL_FIELDS])),
    }
