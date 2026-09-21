"""
test_predictions_endpoint.py

Exercises the FastAPI routes in
app/ml/explainibility_inference/api/main.py in isolation from both the
real model and the real database:

- The predictor dependency (get_predictor) is overridden with a fake
  object exposing the same `.predict(df) -> list[dict]` interface, so
  these tests don't need the real joblib artifact or SHAP at all.
- The DB-reading functions (get_customer_features /
  get_customer_features_batch) are monkeypatched at the point main.py
  imported them, so no MySQL connection is required.

This keeps the API test suite fast and independent of both "is a model
loaded" and "is a database reachable" — same spirit as the
tests/inference suite being independent of Person 5's final model.

Moved here from the top-level tests/api/ (see git history / PR) so all
of Person 6's tests — unit, integration, and API — live under one
app/ml/tests/explainibility_inference_tests/ tree, mirroring the
inference/ and data_access/ subfolders already there.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.ml.explainibility_inference.api.main as main_module
from app.ml.explainibility_inference.api.dependencies import get_predictor
from app.ml.explainibility_inference.data_access.customer_feature_repository import (
    CustomerNotFoundError,
)


class FakePredictor:
    """Stands in for ChurnPredictor: returns one canned record per input row."""

    def predict(self, customer_features: pd.DataFrame) -> list[dict]:
        return [
            {
                "customer_unique_id": customer_id,
                "churn_probability": 0.42,
                "shap_values": {"recency_days": 0.1},
                "reason_codes": ["RC01"],
                "model_version": "v0.1-dummy",
                "scored_at": "2026-01-01T00:00:00+00:00",
            }
            for customer_id in customer_features["customer_unique_id"]
        ]


@pytest.fixture
def client():
    main_module.app.dependency_overrides[get_predictor] = FakePredictor
    yield TestClient(main_module.app)
    main_module.app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_from_features_bypasses_db(client):
    payload = {
        "customers": [
            {
                "customer_unique_id": "C001",
                "features": {"recency_days": 12.0, "frequency": 3.0},
            }
        ]
    }
    response = client.post(f"{main_module.API_PREFIX}/predictions/from-features", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body[0]["customer_unique_id"] == "C001"
    assert body[0]["reason_codes"] == ["RC01"]


def test_predict_one_success(client, monkeypatch):
    fake_df = pd.DataFrame([{"customer_unique_id": "C001", "recency_days": 12.0}])
    monkeypatch.setattr(main_module, "get_customer_features", lambda cid: fake_df)

    response = client.get(f"{main_module.API_PREFIX}/predictions/C001")

    assert response.status_code == 200
    assert response.json()["customer_unique_id"] == "C001"


def test_predict_one_not_found(client, monkeypatch):
    def raise_not_found(customer_unique_id):
        raise CustomerNotFoundError(customer_unique_id)

    monkeypatch.setattr(main_module, "get_customer_features", raise_not_found)

    response = client.get(f"{main_module.API_PREFIX}/predictions/UNKNOWN")

    assert response.status_code == 404


def test_predict_batch_success(client, monkeypatch):
    fake_df = pd.DataFrame(
        [
            {"customer_unique_id": "C001", "recency_days": 12.0},
            {"customer_unique_id": "C002", "recency_days": 30.0},
        ]
    )
    monkeypatch.setattr(main_module, "get_customer_features_batch", lambda ids: fake_df)

    response = client.post(
        f"{main_module.API_PREFIX}/predictions/batch",
        json={"customer_unique_ids": ["C001", "C002"]},
    )

    assert response.status_code == 200
    ids = [record["customer_unique_id"] for record in response.json()]
    assert ids == ["C001", "C002"]


def test_predict_batch_not_found(client, monkeypatch):
    def raise_not_found(ids):
        raise CustomerNotFoundError(ids)

    monkeypatch.setattr(main_module, "get_customer_features_batch", raise_not_found)

    response = client.post(
        f"{main_module.API_PREFIX}/predictions/batch",
        json={"customer_unique_ids": ["UNKNOWN"]},
    )

    assert response.status_code == 404
