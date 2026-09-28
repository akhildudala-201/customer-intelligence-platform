
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from app.api.cohort_trend_api.routers import get_engine, router
from app.api.cohort_trend_api import services as svc

PREFIX = "/api/v1"
MONTHS = pd.date_range("2017-01-01", periods=6, freq="MS")


# ----------------------------------------------------------------------
# Test data in the pipelines' real table formats
# ----------------------------------------------------------------------
def churn_rows():
    monthly = pd.DataFrame({
        "granularity": "monthly",
        "period_date": MONTHS,
        "active_customers": [100, 200, 300, 400, 500, 600],
        "new_customers": [100, 100, 0, 120, 130, 140],
        "repeat_customers": [0, 10, 0, 30, 40, 60],          # month 3: nobody bought -> share None
        "churned_customers": [0, 5, 10, 20, 30, 40],
        "churn_rate": [0.0, 0.025, 0.0333, 0.05, 0.06, 0.0667],
        "retention_rate": [1.0, 0.975, 0.9667, 0.95, 0.94, 0.9333],
        "rolling_churn_rate": [0.0, 0.0125, 0.0194, 0.0361, 0.0478, 0.0589],
        "cumulative_churned": [0, 5, 15, 35, 65, 105],
        "is_censored": [0, 0, 0, 0, 1, 1],                   # MySQL stores bools as 0/1
        "is_forecast": 0,
    })
    weekly = monthly.head(3).assign(granularity="weekly",
                                    period_date=pd.date_range("2017-01-02", periods=3, freq="W-MON"))
    return pd.concat([monthly, weekly], ignore_index=True)


def revenue_rows():
    df = pd.DataFrame({
        "granularity": "monthly",
        "period_date": list(MONTHS) + [pd.Timestamp("2017-07-01")],
        "total_revenue": [1000.0, 1500.0, 1200.0, 1800.0, 2000.0, 2100.0, 9999.0],
        "order_count": [10, 15, 12, 18, 20, 21, 99],
        "unique_customers": [10, 14, 12, 17, 19, 20, 99],
        "avg_order_value": 100.0,
        "avg_revenue_per_user": 105.0,
        "revenue_growth_pct": [np.nan, 50.0, -20.0, 50.0, 11.11, 5.0, 0.0],   # first period has no growth
        "rolling_revenue_sma_short": 1200.0,
        "rolling_revenue_sma_long": 1300.0,
        "cumulative_revenue": [1000.0, 2500.0, 3700.0, 5500.0, 7500.0, 9600.0, 0.0],
        "is_forecast": [0, 0, 0, 0, 0, 0, 1],                # last row is a forecast -> excluded
    })
    return df


def combined_rows():
    # Older pipeline run: no is_censored column (the API must still work)
    rev = revenue_rows().iloc[:6].drop(columns=["rolling_revenue_sma_short",
                                                "rolling_revenue_sma_long", "cumulative_revenue"])
    ch = churn_rows().query("granularity == 'monthly'").drop(columns=["is_censored", "cumulative_churned"])
    return rev.merge(ch, on=["granularity", "period_date", "is_forecast"])


def cohort_rows():
    """Dec 2016 has 12 months (M0-M11), Feb 2017 has 4, Apr 2017 has 3.
    Alphabetical order would be Apr, Dec, Feb; the API must return Dec, Feb, Apr."""
    rows = []
    for label, n_months, final_churn, size in [("Dec 2016", 12, 80.0, 500),
                                               ("Feb 2017", 4, 75.5, 300),
                                               ("Apr 2017", 3, np.nan, 200)]:
        for m in range(n_months):
            rows.append({
                "cohort": label,
                "cohort_size": size,
                "relative_month": f"M{m}",
                "retention_rate (%)": 100.0 if m == 0 else round(10 - m * 0.5, 2),
                "repeat_purchase_rate (%)": 0.0 if m == 0 else 1.0,
                "cumulative_repeat_purchase_rate (%)": float(m),
                "final_churn_rate (%)": final_churn,
                "monthly_churn_rate (%)": np.nan if m == 0 else 2.0,   # NULL when nobody at risk
                "cumulative_churn_rate (%)": m * 2.0,
                "cumulative_average_revenue": 150.0 + m * 10,
                "generated_date": "2026-09-20",
            })
    # Stored in scrambled order to prove the API sorts by date and month number
    return pd.DataFrame(rows).sample(frac=1, random_state=0)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    churn_rows().to_sql(svc.CHURN_TABLE, engine, index=False)
    revenue_rows().to_sql(svc.REVENUE_TABLE, engine, index=False)
    combined_rows().to_sql(svc.COMBINED_TABLE, engine, index=False)
    cohort_rows().to_sql(svc.COHORT_TABLE, engine, index=False)
    return engine


def make_client(engine):
    app = FastAPI()
    app.include_router(router, prefix=PREFIX)
    app.dependency_overrides[get_engine] = lambda: engine
    return TestClient(app)


@pytest.fixture
def client(db):
    return make_client(db)


@pytest.fixture(autouse=True)
def fresh_cache():
    svc.clear_cache()
    yield
    svc.clear_cache()


def get(client, path, **params):
    return client.get(PREFIX + path, params=params)


# ----------------------------------------------------------------------
# Trend endpoints
# ----------------------------------------------------------------------
class TestChurnTrend:
    def test_defaults_to_monthly_sorted(self, client):
        body = get(client, "/churn/trend").json()
        assert body["granularity"] == "monthly"
        assert body["count"] == 6
        dates = [p["period_date"] for p in body["data"]]
        assert dates == sorted(dates) and dates[0] == "2017-01-01"

    def test_granularity_filter(self, client):
        body = get(client, "/churn/trend", granularity="weekly").json()
        assert body["count"] == 3 and body["granularity"] == "weekly"

    def test_censored_flag_is_boolean(self, client):
        flags = [p["is_censored"] for p in get(client, "/churn/trend").json()["data"]]
        assert flags == [False, False, False, False, True, True]

    def test_date_range(self, client):
        body = get(client, "/churn/trend", start_date="2017-02-01", end_date="2017-04-01").json()
        assert [p["period_date"] for p in body["data"]] == ["2017-02-01", "2017-03-01", "2017-04-01"]

    def test_start_after_end_is_400(self, client):
        assert get(client, "/churn/trend", start_date="2017-05-01", end_date="2017-01-01").status_code == 400

    @pytest.mark.parametrize("params", [{"granularity": "yearly"}, {"start_date": "not-a-date"}])
    def test_invalid_params_are_422(self, client, params):
        assert get(client, "/churn/trend", **params).status_code == 422

    def test_range_with_no_rows_is_empty_200(self, client):
        r = get(client, "/churn/trend", start_date="2030-01-01")
        assert r.status_code == 200 and r.json()["count"] == 0


class TestRevenueTrend:
    def test_excludes_forecast_rows_and_keeps_null_growth(self, client):
        data = get(client, "/revenue/trend").json()["data"]
        assert len(data) == 6                                  # 2017-07 forecast row dropped
        assert data[0]["revenue_growth_pct"] is None           # NULL, not 0
        assert data[1]["total_revenue"] == 1500.0


class TestCombinedTrend:
    def test_missing_is_censored_column_returns_null(self, client):
        data = get(client, "/trends/combined").json()["data"]
        assert len(data) == 6
        assert all(p["is_censored"] is None for p in data)
        assert data[0]["total_revenue"] == 1000.0 and data[0]["churn_rate"] == 0.0


class TestNewVsRepeat:
    def test_repeat_share(self, client):
        data = get(client, "/trends/new-vs-repeat").json()["data"]
        assert data[1]["repeat_share_pct"] == pytest.approx(10 / 110 * 100, abs=0.01)
        assert data[2]["repeat_share_pct"] is None             # 0 new + 0 repeat


# ----------------------------------------------------------------------
# Cohort endpoints
# ----------------------------------------------------------------------
class TestCohortRetention:
    def test_matrix_shape_and_ordering(self, client):
        body = get(client, "/customers/cohort-retention").json()
        assert body["cohorts"] == ["2016-12", "2017-02", "2017-04"]     # by date, not alphabet
        assert body["labels"] == ["Dec 2016", "Feb 2017", "Apr 2017"]
        assert body["months"] == [f"M{m}" for m in range(12)]           # M10 after M9
        assert body["generated_date"] == "2026-09-20"

    def test_unreached_months_are_null_not_zero(self, client):
        values = get(client, "/customers/cohort-retention").json()["values"]
        assert values[0][11] == pytest.approx(4.5)             # Dec 2016 reached M11
        assert values[2][:3] == [100.0, 9.5, 9.0]              # Apr 2017 has M0-M2
        assert values[2][3:] == [None] * 9                     # ...and nothing after

    def test_cohort_range_filter(self, client):
        body = get(client, "/customers/cohort-retention", cohort_from="2017-01", cohort_to="2017-03").json()
        assert body["cohorts"] == ["2017-02"]
        assert body["months"] == ["M0", "M1", "M2", "M3"]

    def test_bad_cohort_format_is_422(self, client):
        assert get(client, "/customers/cohort-retention", cohort_from="Feb 2017").status_code == 422


class TestCohortSeries:
    def test_churn(self, client):
        body = get(client, "/cohort/churn").json()
        assert body["count"] == 3
        dec = body["data"][0]
        assert dec["cohort"] == "2016-12" and dec["final_churn_rate_pct"] == 80.0
        assert dec["months"][0] == {"month": "M0", "monthly_churn_rate_pct": None,
                                    "cumulative_churn_rate_pct": 0.0}
        assert [m["month"] for m in dec["months"]][9:] == ["M9", "M10", "M11"]
        assert body["data"][2]["final_churn_rate_pct"] is None           # NULL in the table

    def test_revenue(self, client):
        feb = get(client, "/cohort/revenue").json()["data"][1]
        assert feb["cohort"] == "2017-02"
        assert feb["latest_cumulative_average_revenue"] == 180.0         # M3 value
        assert len(feb["months"]) == 4

    def test_repeat_purchase(self, client):
        apr = get(client, "/cohort/repeat-purchase").json()["data"][2]
        assert apr["latest_cumulative_repeat_purchase_rate_pct"] == 2.0
        assert apr["months"][1] == {"month": "M1", "repeat_purchase_rate_pct": 1.0,
                                    "cumulative_repeat_purchase_rate_pct": 1.0}


class TestCohortSummary:
    def test_one_row_per_cohort(self, client):
        data = get(client, "/cohort/summary").json()["data"]
        dec, feb, apr = data
        assert dec == {
            "cohort": "2016-12", "label": "Dec 2016", "cohort_size": 500, "months_observed": 12,
            "m1_retention_pct": 9.5, "m3_retention_pct": 8.5, "final_churn_rate_pct": 80.0,
            "cumulative_repeat_purchase_rate_pct": 11.0, "cumulative_average_revenue": 260.0,
        }
        assert apr["m3_retention_pct"] is None                 # Apr 2017 only reached M2

    def test_works_without_cohort_size_column(self, tmp_path):
        engine = create_engine(f"sqlite:///{tmp_path / 'no_size.db'}")
        cohort_rows().drop(columns="cohort_size").to_sql(svc.COHORT_TABLE, engine, index=False)
        data = get(make_client(engine), "/cohort/summary").json()["data"]
        assert all(row["cohort_size"] is None for row in data)


class TestCohortDetail:
    def test_found(self, client):
        body = get(client, "/cohort/2017-02").json()
        assert body["label"] == "Feb 2017" and body["cohort_size"] == 300
        assert [m["month"] for m in body["months"]] == ["M0", "M1", "M2", "M3"]
        assert body["months"][0]["retention_rate_pct"] == 100.0

    def test_unknown_cohort_is_404(self, client):
        assert get(client, "/cohort/2030-01").status_code == 404

    @pytest.mark.parametrize("bad", ["2017-13", "Mar%202017", "2017-3"])
    def test_bad_format_is_422(self, client, bad):
        assert client.get(f"{PREFIX}/cohort/{bad}").status_code == 422

    def test_fixed_routes_not_treated_as_cohort(self, client):
        assert get(client, "/cohort/summary").json()["count"] == 3


# ----------------------------------------------------------------------
# Errors and caching
# ----------------------------------------------------------------------
class TestInfrastructure:
    @pytest.mark.parametrize("path", ["/churn/trend", "/cohort/summary", "/cohort/2017-02"])
    def test_missing_table_is_503(self, tmp_path, path):
        empty = make_client(create_engine(f"sqlite:///{tmp_path / 'empty.db'}"))
        r = get(empty, path)
        assert r.status_code == 503 and "pipeline" in r.json()["detail"]

    def test_second_request_served_from_cache(self, client, db, monkeypatch):
        monkeypatch.setattr(svc, "CACHE_TTL", 300)
        assert get(client, "/cohort/summary").status_code == 200
        with db.begin() as conn:                               # table gone...
            conn.execute(text(f"DROP TABLE {svc.COHORT_TABLE}"))
        assert get(client, "/cohort/summary").status_code == 200   # ...still served from cache
        svc.clear_cache()
        assert get(client, "/cohort/summary").status_code == 503   # cache cleared -> real read

    def test_cache_disabled_reads_every_time(self, client, db, monkeypatch):
        monkeypatch.setattr(svc, "CACHE_TTL", 0)
        assert get(client, "/cohort/summary").status_code == 200
        with db.begin() as conn:
            conn.execute(text(f"DROP TABLE {svc.COHORT_TABLE}"))
        assert get(client, "/cohort/summary").status_code == 503
