"""
test_generate_predictions_table.py

Monkeypatches get_all_customer_features, ChurnPredictor, and
DataFrame.to_sql so this test runs without a real DB connection or a
real model artifact, while still exercising the real
records -> DataFrame -> to_sql shaping logic.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import app.ml.explainibility_interface.inference.generate_predictions_table as gpt
from app.ml.explainibility_interface.inference.feature_contract import (
    REQUIRED_FEATURES,
)


class _FakePredictor:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, customer_features: pd.DataFrame) -> list[dict]:
        return [
            {
                "customer_unique_id": row["customer_unique_id"],
                "churn_probability": 0.42,
                "shap_values": {"monetary_value": 0.1, "avg_review_score": -0.05},
                "reason_codes": ["RC01"],
                "model_version": "test-version",
                "scored_at": "2026-09-21T00:00:00+00:00",
            }
            for _, row in customer_features.iterrows()
        ]


def test_generate_predictions_table_writes_expected_columns(monkeypatch):
    feature_values = {feature: 1.0 for feature in REQUIRED_FEATURES}
    fake_features = pd.DataFrame(
        [
            {"customer_unique_id": "C001", **feature_values},
            {"customer_unique_id": "C002", **feature_values},
        ]
    )
    monkeypatch.setattr(gpt, "get_all_customer_features", lambda: fake_features)
    monkeypatch.setattr(gpt, "ChurnPredictor", _FakePredictor)

    captured = {}

    def _fake_to_sql(self, name, con, if_exists, index):
        captured["name"] = name
        captured["if_exists"] = if_exists
        captured["index"] = index
        captured["df"] = self

    monkeypatch.setattr(pd.DataFrame, "to_sql", _fake_to_sql)

    # app.Database.database is imported lazily inside the function under
    # test; stub it out before it's imported so no real DB_* env vars or
    # connection are needed.
    import sys
    import types

    fake_database_module = types.ModuleType("app.Database.database")
    setattr(fake_database_module, "engine", object())
    monkeypatch.setitem(sys.modules, "app.Database.database", fake_database_module)

    result = gpt.generate_predictions_table()

    assert captured["name"] == "churn_predictions"
    assert captured["if_exists"] == "replace"
    assert captured["index"] is False
    assert list(result.columns) == [
        "customer_unique_id",
        "churn_probability",
        "shap_values",
        "reason_codes",
        "model_version",
        "scored_at",
    ]
    assert result["customer_unique_id"].tolist() == ["C001", "C002"]
    # shap_values/reason_codes must be JSON strings, not raw dict/list objects
    assert json.loads(result.iloc[0]["shap_values"]) == {
        "monetary_value": 0.1,
        "avg_review_score": -0.05,
    }
    assert json.loads(result.iloc[0]["reason_codes"]) == ["RC01"]


def test_generate_predictions_table_raises_when_no_customers(monkeypatch):
    def _raise_no_customers():
        from app.ml.explainibility_interface.data_access.customer_feature_repository import (
            CustomerNotFoundError,
        )

        raise CustomerNotFoundError("no rows")

    monkeypatch.setattr(gpt, "get_all_customer_features", _raise_no_customers)

    from app.ml.explainibility_interface.data_access.customer_feature_repository import (
        CustomerNotFoundError,
    )

    with pytest.raises(CustomerNotFoundError):
        gpt.generate_predictions_table()
