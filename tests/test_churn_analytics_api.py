from __future__ import annotations

import pytest

from app.api.churn_analytics import router as churn_router


def test_churn_summary_and_probability_distribution(monkeypatch):
    monkeypatch.setattr(
        churn_router.repo,
        "churn_summary",
        lambda threshold: {
            "total_customers": 4,
            "avg_churn_probability": 0.4,
            "predicted_churn_customers": 2,
            "refreshed_at": None,
        },
    )
    summary = churn_router.churn_summary()
    assert summary["total_customers"] == 4
    assert summary["predicted_churn_rate"] == 0.5

    monkeypatch.setattr(
        churn_router.repo,
        "fetch_all",
        lambda *_args, **_kwargs: [
            {"churn_probability": 0.0},
            {"churn_probability": 0.54},
            {"churn_probability": 1.0},
        ],
    )
    distribution = churn_router.probability_distribution()
    assert distribution["total_customers"] == 3
    assert [bucket["count"] for bucket in distribution["bins"]] == [1, 0, 0, 0, 0, 1, 0, 0, 0, 1]


def test_explanation_reads_stored_values_and_reason_messages(monkeypatch):
    monkeypatch.setattr(
        churn_router.repo,
        "prediction_for_customer",
        lambda _customer_id: {
            "customer_unique_id": "C001",
            "churn_probability": 0.73,
            "shap_values": '{"avg_review_score": 0.2}',
            "reason_codes": '["RC01B"]',
            "model_version": "test-model",
            "scored_at": "2026-09-29T00:00:00+00:00",
        },
    )
    body = churn_router.customer_explanation("C001")
    assert body["shap_values"] == {"avg_review_score": 0.2}
    assert body["reason_codes"][0]["code"] == "RC01B"
    assert "negative or lukewarm reviews" in body["reason_codes"][0]["message"]


def test_predict_scores_requested_customer(monkeypatch):
    import pandas as pd

    class FakePredictor:
        def predict(self, features):
            return [{
                "customer_unique_id": features.iloc[0]["customer_unique_id"],
                "churn_probability": 0.5,
                "shap_values": {},
                "reason_codes": [],
                "model_version": "test-model",
                "scored_at": "2026-09-29T00:00:00+00:00",
            }]

    monkeypatch.setattr(
        churn_router,
        "_load_customer_features",
        lambda customer_id: pd.DataFrame([{"customer_unique_id": customer_id}]),
    )
    result = churn_router.predict_customer(
        churn_router.PredictRequest(customer_unique_id="C001"), FakePredictor()
    )
    assert result["customer_unique_id"] == "C001"


def test_reason_code_summary_includes_unused_configured_codes(monkeypatch):
    monkeypatch.setattr(churn_router.repo, "prediction_values", lambda: [{"reason_codes": '["RC01B"]'}])
    result = churn_router.reason_codes_summary()
    summary = {item["reason_code"]: item for item in result["reason_codes"]}
    assert summary["RC01B"]["count"] == 1
    assert summary["RC01"]["count"] == 0


def test_all_requested_paths_are_registered():
    paths = {route.path for route in churn_router.router.routes}
    assert {
        "/churn/summary",
        "/churn/top-features",
        "/churn/feature-importance-global",
        "/predict",
        "/members/{id}/risk",
        "/data/last-refresh",
        "/churn/probability-distribution",
        "/churn/reason-codes/summary",
        "/customers/{id}/explanation",
        "/data/tables",
    } <= paths