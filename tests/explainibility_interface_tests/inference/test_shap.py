"""
test_shap.py

Covers: SHAP output structure and SHAP JSON serialization, for both single
and multi-customer input. These tests only exercise ChurnShapExplainer and
ChurnModelAdapter, not the full predict() pipeline (see
test_inference_pipeline.py for that).
"""

from __future__ import annotations

import json

from app.ml.explainibility_interface.explainability.shap_explainer import ChurnShapExplainer
from app.ml.explainibility_interface.inference.feature_contract import REQUIRED_FEATURES
from app.ml.explainibility_interface.inference.model_adapter import ChurnModelAdapter
from app.ml.explainibility_interface.inference.model_loader import load_model
from app.ml.explainibility_interface.inference.predict import DEFAULT_MODEL_PATH


def _build_explainer() -> ChurnShapExplainer:
    raw_model = load_model(DEFAULT_MODEL_PATH)
    adapter = ChurnModelAdapter(raw_model)
    return ChurnShapExplainer(adapter.get_shap_model(), feature_names=REQUIRED_FEATURES)


def test_shap_output_single_customer(sample_customer_df):
    explainer = _build_explainer()
    X = sample_customer_df[REQUIRED_FEATURES]

    result = explainer.explain(X)

    assert isinstance(result, list)
    assert len(result) == 1
    assert set(result[0].keys()) == set(REQUIRED_FEATURES)


def test_shap_output_batch(sample_batch_df):
    explainer = _build_explainer()
    X = sample_batch_df[REQUIRED_FEATURES]

    result = explainer.explain(X)

    assert len(result) == len(sample_batch_df)
    for row in result:
        assert set(row.keys()) == set(REQUIRED_FEATURES)


def test_shap_values_are_plain_python_floats(sample_customer_df):
    explainer = _build_explainer()
    X = sample_customer_df[REQUIRED_FEATURES]

    result = explainer.explain(X)

    for value in result[0].values():
        assert isinstance(value, float)


def test_shap_json_serializable(sample_batch_df):
    """SHAP output must be directly json.dumps-able with no numpy leakage."""
    explainer = _build_explainer()
    X = sample_batch_df[REQUIRED_FEATURES]

    result = explainer.explain(X)

    serialized = json.dumps(result)  # raises TypeError if not serializable
    assert isinstance(serialized, str)


def test_shap_column_order_mismatch_raises(sample_customer_df):
    explainer = _build_explainer()
    # Deliberately wrong column order.
    reversed_features = list(reversed(REQUIRED_FEATURES))
    X = sample_customer_df[reversed_features]

    import pytest

    with pytest.raises(ValueError):
        explainer.explain(X)
