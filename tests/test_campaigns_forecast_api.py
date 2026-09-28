from __future__ import annotations

import pytest

EXPECTED_PATHS = [
    "/campaigns/active",
    "/campaigns/recommendations",
    "/campaigns/reason-codes",
    "/campaigns/rules",
    "/campaigns/by-segment",
    "/campaigns/{campaign_name}/customers",
    "/campaigns/evaluate",
    "/churn/delivery-correlation",
    "/churn/review-correlation",
    "/forecast/churn",
    "/forecast/revenue",
    "/forecast/accuracy",
]

DELIVERY_ORDER = [
    "7+ days early",
    "On time / up to 7 days early",
    "1-3 days late",
    "4-7 days late",
    "8+ days late",
]
REVIEW_ORDER = [
    "Poor (<2)",
    "Below average (2-3)",
    "Average (3-4)",
    "Good (4-5)",
    "Perfect (5)",
    "No review",
]


# ----------------------------------------------------------------------
# fixtures / helpers
# ----------------------------------------------------------------------
class Api:
    """Tiny wrapper that adds the API prefix (default /api/v1) to every path."""

    def __init__(self, client, app, prefix: str):
        self._client, self.app, self.prefix = client, app, prefix

    def get(self, path: str, **params):
        return self._client.get(self.prefix + path, params=params)

    def post(self, path: str, json: dict):
        return self._client.post(self.prefix + path, json=json)


@pytest.fixture(scope="module")
def api() -> Api:
    try:
        from app.api.main import API_PREFIX, app
    except RuntimeError as exc:  # app/Database/database.py raises if DB_* vars are missing
        pytest.skip(f"Database env vars missing (check .env): {exc}")
    from fastapi.testclient import TestClient

    return Api(TestClient(app), app, API_PREFIX)


def ok(resp, skip_on=(503,)):
    """Return the JSON of a 200 response; skip if the DB/table isn't available."""
    if resp.status_code in skip_on:
        pytest.skip(f"Data not available ({resp.status_code}): {resp.text[:200]}")
    assert resp.status_code == 200, resp.text
    return resp.json()


def expect_status(resp, code: int):
    if resp.status_code == 503:
        pytest.skip(f"Database/table unavailable: {resp.text[:200]}")
    assert resp.status_code == code, resp.text


@pytest.fixture(scope="module")
def active(api):
    return ok(api.get("/campaigns/active"))


@pytest.fixture(scope="module")
def sample_items(api, active):
    data = ok(api.get("/campaigns/recommendations", page_size=20))
    assert data["items"], "customer_campaign_recommendations is empty"
    return data["items"]


# ----------------------------------------------------------------------
# routes registered
# ----------------------------------------------------------------------
def test_all_p5_routes_are_registered(api):
    paths = api.app.openapi()["paths"]
    for path in EXPECTED_PATHS:
        assert api.prefix + path in paths, f"Missing route: {api.prefix + path}"


# ----------------------------------------------------------------------
# rulebook endpoints (no database needed)
# ----------------------------------------------------------------------
def test_reason_codes(api):
    data = ok(api.get("/campaigns/reason-codes"))
    assert len(data) >= 1
    for item in data:
        assert item["reason_code"] and item["explanation"]


def test_rules(api):
    data = ok(api.get("/campaigns/rules"))
    assert data["total_rules"] == len(data["rules"]) > 0
    assert data["default_rule"]["campaign_name"]
    for rule in data["rules"]:
        assert 1 <= rule["campaign_priority"] <= 5
        assert rule["risk_tier"] in data["risk_tiers"]
        assert rule["segment_label"] in data["segments"]
        assert rule["value_tier"] in data["value_tiers"]
        assert rule["explanation"]


def test_evaluate_what_if_matches_rulebook(api):
    rule = ok(api.get("/campaigns/rules"))["rules"][0]
    body = {k: rule[k] for k in ("risk_tier", "segment_label", "value_tier")}
    data = ok(api.post("/campaigns/evaluate", body))
    assert data["source"] == "rules"
    assert data["matched_explicit_rule"] is True
    assert data["campaign_name"] == rule["campaign_name"]
    assert data["campaign_priority"] == rule["campaign_priority"]
    assert data["reason_code"] == rule["reason_code"]
    assert data["reason"]


def test_evaluate_unknown_profile_falls_back_to_default_rule(api):
    default = ok(api.get("/campaigns/rules"))["default_rule"]
    body = {"risk_tier": "No Such Tier", "segment_label": "No Such Segment", "value_tier": "No Such Value"}
    data = ok(api.post("/campaigns/evaluate", body))
    assert data["matched_explicit_rule"] is False
    assert data["campaign_name"] == default["campaign_name"]
    assert data["reason_code"] == default["reason_code"]


@pytest.mark.parametrize(
    "body",
    [{}, {"risk_tier": "High Risk"}, {"risk_tier": "High Risk", "segment_label": "x"}],
)
def test_evaluate_requires_id_or_full_profile(api, body):
    assert api.post("/campaigns/evaluate", body).status_code == 422


# ----------------------------------------------------------------------
# campaigns (database)
# ----------------------------------------------------------------------
def test_active_campaigns_are_consistent(active):
    campaigns = active["campaigns"]
    assert active["active_campaigns"] == len(campaigns) > 0
    assert active["customers_targeted"] == sum(c["customer_count"] for c in campaigns)
    high = sum(c["customer_count"] for c in campaigns if c["campaign_priority"] <= 2)
    assert active["high_priority_customers"] == high
    assert high <= active["customers_targeted"]
    priorities = [c["campaign_priority"] for c in campaigns]
    assert priorities == sorted(priorities), "campaigns should be ordered by priority"


def test_every_stored_campaign_exists_in_rulebook(api, active):
    rules = ok(api.get("/campaigns/rules"))
    known = {r["campaign_name"] for r in rules["rules"]} | {rules["default_rule"]["campaign_name"]}
    stored = {c["campaign_name"] for c in active["campaigns"]}
    assert stored <= known, f"Campaigns in DB but not in YAML (rerun the pipeline?): {stored - known}"


def test_recommendations_totals_and_summary(api, active):
    data = ok(api.get("/campaigns/recommendations", page_size=10))
    assert data["total"] == active["customers_targeted"]
    assert len(data["items"]) <= 10
    assert sum(row["customer_count"] for row in data["summary"]) == data["total"]
    for item in data["items"]:
        assert item["customer_unique_id"] and item["campaign_name"] and item["reason_code"]


def test_recommendations_filter_by_campaign(api, active):
    top = active["campaigns"][0]
    data = ok(api.get("/campaigns/recommendations", campaign_name=top["campaign_name"], page_size=5))
    assert data["total"] == top["customer_count"]
    assert all(i["campaign_name"] == top["campaign_name"] for i in data["items"])


def test_recommendations_filter_by_priority(api, active):
    expected = sum(c["customer_count"] for c in active["campaigns"] if c["campaign_priority"] == 1)
    data = ok(api.get("/campaigns/recommendations", campaign_priority=1, page_size=5))
    assert data["total"] == expected
    assert all(i["campaign_priority"] == 1 for i in data["items"])


def test_recommendations_pagination(api, active):
    if active["customers_targeted"] <= 5:
        pytest.skip("Not enough rows to test pagination")
    page1 = ok(api.get("/campaigns/recommendations", page=1, page_size=5))["items"]
    page2 = ok(api.get("/campaigns/recommendations", page=2, page_size=5))["items"]
    ids1 = {i["customer_unique_id"] for i in page1}
    ids2 = {i["customer_unique_id"] for i in page2}
    assert len(page1) == len(page2) == 5
    assert ids1.isdisjoint(ids2)


@pytest.mark.parametrize("params", [{"page_size": 0}, {"page_size": 501}, {"page": 0}, {"campaign_priority": 9}])
def test_recommendations_rejects_bad_params(api, params):
    assert api.get("/campaigns/recommendations", **params).status_code == 422


def test_by_segment_adds_up(api, active):
    data = ok(api.get("/campaigns/by-segment"))
    assert data, "no segments returned"
    assert sum(s["total_customers"] for s in data) == active["customers_targeted"]
    for seg in data:
        assert seg["total_customers"] == sum(c["customer_count"] for c in seg["campaigns"])
    totals = [s["total_customers"] for s in data]
    assert totals == sorted(totals, reverse=True)


def test_campaign_customers_drill_down(api, active):
    top = active["campaigns"][0]
    data = ok(api.get(f"/campaigns/{top['campaign_name']}/customers", page_size=25))
    assert data["campaign_name"] == top["campaign_name"]
    assert data["total"] == top["customer_count"]
    assert 0 < len(data["items"]) <= 25
    assert all(i["campaign_name"] == top["campaign_name"] for i in data["items"])
    probs = [i["churn_probability"] for i in data["items"] if i["churn_probability"] is not None]
    assert all(0 <= p <= 1 for p in probs)
    assert probs == sorted(probs, reverse=True), "should be sorted by churn probability, highest first"


def test_campaign_customers_unknown_campaign_is_404(api):
    expect_status(api.get("/campaigns/This Campaign Does Not Exist/customers"), 404)


def test_evaluate_real_customer_matches_stored_recommendation(api, sample_items):
    item = sample_items[0]
    data = ok(api.post("/campaigns/evaluate", {"customer_unique_id": item["customer_unique_id"]}))
    assert data["source"] == "database"
    assert data["customer_unique_id"] == item["customer_unique_id"]
    for key in ("campaign_name", "campaign_priority", "reason_code", "risk_tier", "segment_label", "value_tier"):
        assert data[key] == item[key]
    assert data["reason"]


def test_evaluate_unknown_customer_is_404(api):
    expect_status(api.post("/campaigns/evaluate", {"customer_unique_id": "does-not-exist"}), 404)


# ----------------------------------------------------------------------
# churn correlations (database)
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "path, dimension, order",
    [
        ("/churn/delivery-correlation", "delivery_delay", DELIVERY_ORDER),
        ("/churn/review-correlation", "review_score", REVIEW_ORDER),
    ],
)
def test_correlation_structure(api, path, dimension, order):
    data = ok(api.get(path))
    assert data["dimension"] == dimension
    assert data["threshold"] == 0.5
    buckets = data["buckets"]
    assert buckets, "no buckets returned - are customer_features and churn_predictions populated?"
    assert data["overall_customers"] == sum(b["customer_count"] for b in buckets) > 0
    for b in buckets:
        assert b["bucket"] in order, f"Unexpected bucket: {b['bucket']}"
        assert 0 <= b["avg_churn_probability"] <= 1
        assert 0 <= b["predicted_churn_rate_pct"] <= 100
    positions = [order.index(b["bucket"]) for b in buckets]
    assert positions == sorted(positions), "buckets should come back in logical order"


@pytest.mark.parametrize("path", ["/churn/delivery-correlation", "/churn/review-correlation"])
def test_correlation_threshold_parameter(api, path):
    everyone = ok(api.get(path, threshold=0))
    assert everyone["overall_predicted_churn_rate_pct"] == pytest.approx(100.0)
    assert all(b["predicted_churn_rate_pct"] == pytest.approx(100.0) for b in everyone["buckets"])
    default = ok(api.get(path))
    strict = ok(api.get(path, threshold=0.9))
    assert strict["overall_predicted_churn_rate_pct"] <= default["overall_predicted_churn_rate_pct"]


@pytest.mark.parametrize("path", ["/churn/delivery-correlation", "/churn/review-correlation"])
@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_correlation_rejects_bad_threshold(api, path, threshold):
    assert api.get(path, threshold=threshold).status_code == 422


# ----------------------------------------------------------------------
# forecasting (database)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("metric", ["churn", "revenue"])
def test_forecast_shape_and_confidence_interval(api, metric):
    data = ok(api.get(f"/forecast/{metric}", horizon=6), skip_on=(503, 404))
    assert data["confidence_level"] == 0.95
    assert data["available_periods"] >= 1
    assert data["horizon"] == len(data["data"]) == min(6, data["available_periods"])
    dates = [p["period_date"] for p in data["data"]]
    assert dates == sorted(dates), "forecast periods should be in chronological order"
    for p in data["data"]:
        assert p["forecast"] is not None
        assert p["ci_lower"] <= p["forecast"] <= p["ci_upper"], p
        if metric == "churn":
            assert 0 <= p["ci_lower"] and p["ci_upper"] <= 1, "churn rate must stay between 0 and 1"
        else:
            assert p["ci_lower"] >= 0, "revenue can't be negative"


@pytest.mark.parametrize("metric", ["churn", "revenue"])
def test_forecast_horizon_is_respected(api, metric):
    one = ok(api.get(f"/forecast/{metric}", horizon=1), skip_on=(503, 404))
    assert len(one["data"]) == 1
    everything = ok(api.get(f"/forecast/{metric}", horizon=60), skip_on=(503, 404))
    assert len(everything["data"]) == everything["available_periods"]


@pytest.mark.parametrize("metric", ["churn", "revenue"])
@pytest.mark.parametrize("horizon", [0, 61])
def test_forecast_rejects_bad_horizon(api, metric, horizon):
    assert api.get(f"/forecast/{metric}", horizon=horizon).status_code == 422


def test_forecast_accuracy_matches_forecasts(api):
    acc = ok(api.get("/forecast/accuracy"))
    for metric in ("revenue", "churn"):
        entry = acc[metric]
        assert entry is not None, f"forecast_{metric}_trend is empty - run the forecasting step"
        assert entry["model_used"]
        assert entry["holdout_mape_pct"] is None or entry["holdout_mape_pct"] >= 0
        forecast = ok(api.get(f"/forecast/{metric}", horizon=60), skip_on=(503, 404))
        assert entry["forecast_periods"] == forecast["available_periods"]
        assert entry["model_used"] == forecast["model_used"]
