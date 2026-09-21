"""
test_model_adapter.py

Covers the model-artifact handling in model_adapter.py:

1. The real artifact is saved as {"model": ..., "feature_cols": [...],
   ...}, not a bare model (see save_model() in train_lightgbm_model.py) —
   ChurnModelAdapter must unwrap it.
2. check_contract_matches_model() must be able to read the trained
   feature list back out of that same dict shape, not just off a bare
   model's .feature_name_ attribute.
3. positive_class inversion — see model_adapter.py's docstring.

Uses a lightweight fake model instead of a real LightGBM one — these
tests are about the unwrapping/inversion logic, not about LightGBM itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.explainibility_inference.inference.feature_contract import REQUIRED_FEATURES, check_contract_matches_model
from app.ml.explainibility_inference.inference.model_adapter import ChurnModelAdapter


class _FakeModel:
    """Minimal stand-in for a fitted sklearn/LightGBM classifier."""

    def predict_proba(self, X):
        # One row per input row, columns = [P(no churn), P(churn)].
        n = len(X)
        return np.column_stack([np.full(n, 0.7), np.full(n, 0.3)])


@pytest.fixture
def sample_X(sample_customer_df):
    return sample_customer_df[REQUIRED_FEATURES]


def test_adapter_accepts_bare_model(sample_X):
    """A plain fitted model, not a dict."""
    adapter = ChurnModelAdapter(_FakeModel())

    proba = adapter.predict_probability(sample_X)

    assert proba.shape == (len(sample_X),)
    assert np.allclose(proba, 0.3)


def test_adapter_unwraps_dict_artifact(sample_X):
    """The real training pipeline's artifact shape: {"model": ..., ...}."""
    artifact = {
        "model": _FakeModel(),
        "feature_cols": REQUIRED_FEATURES,
        "operating_point": {"mode": "rate", "value": 0.05},
        "positive_class": 1,
    }

    adapter = ChurnModelAdapter(artifact)

    # predict_probability and get_shap_model must work against the
    # unwrapped model, not the dict itself.
    proba = adapter.predict_probability(sample_X)
    assert np.allclose(proba, 0.3)
    assert isinstance(adapter.get_shap_model(), _FakeModel)


def test_adapter_dict_without_model_key_treated_as_model(sample_X):
    """A dict that isn't the artifact shape (no "model" key) should NOT be
    unwrapped — avoids silently mishandling an unrelated dict-like model
    object."""
    not_an_artifact = {"predict_proba": "not callable, just checking no unwrap happens"}

    adapter = ChurnModelAdapter(not_an_artifact)

    # No unwrapping happened - the dict itself is treated as "the model".
    assert adapter.get_shap_model() is not_an_artifact


def test_adapter_does_not_invert_when_positive_class_is_churn(sample_X):
    """positive_class == 1 (churned, per build_churn_label.py) -> column 1
    of predict_proba is already P(churn); no inversion."""
    artifact = {"model": _FakeModel(), "feature_cols": REQUIRED_FEATURES, "positive_class": 1}

    proba = ChurnModelAdapter(artifact).predict_probability(sample_X)

    assert np.allclose(proba, 0.3)  # _FakeModel's column 1 is 0.3


def test_adapter_inverts_when_positive_class_is_not_churn(sample_X):
    """positive_class == 0 (retained) -> training targeted the WRONG label
    (see prepare_data()'s auto-minority-detection). Column 1 of
    predict_proba is P(retained), so churn probability must be inverted."""
    artifact = {"model": _FakeModel(), "feature_cols": REQUIRED_FEATURES, "positive_class": 0}

    proba = ChurnModelAdapter(artifact).predict_probability(sample_X)

    assert np.allclose(proba, 0.7)  # 1 - 0.3


def test_check_contract_matches_model_reads_dict_feature_cols(capsys):
    """check_contract_matches_model() must be able to pull the trained
    column list from artifact["feature_cols"] directly, not just from a
    bare model's .feature_name_ attribute."""
    mismatched_artifact = {
        "model": _FakeModel(),
        "feature_cols": ["avg_order_value", "delivered_rate"],  # deliberately != REQUIRED_FEATURES
    }

    check_contract_matches_model(mismatched_artifact)

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "avg_order_value" in captured.out


def test_check_contract_matches_model_silent_when_matching(capsys):
    matching_artifact = {"model": _FakeModel(), "feature_cols": REQUIRED_FEATURES}

    check_contract_matches_model(matching_artifact)

    captured = capsys.readouterr()
    assert "WARNING" not in captured.out


def test_decision_threshold_uses_calibrated_artifacts_real_value():
    """A calibrated artifact's own re-tuned threshold must be used as-is,
    not silently replaced with a hardcoded 0.5."""
    artifact = {
        "base_model": _FakeModel(),
        "calibrator": None,
        "method": "none",
        "threshold": 0.67,
        "model_name": "LightGBM",
        "feature_cols": REQUIRED_FEATURES,
    }

    adapter = ChurnModelAdapter(artifact)

    assert adapter.decision_threshold == 0.67


def test_decision_threshold_falls_back_for_rate_based_operating_point():
    """An uncalibrated artifact's operating_point can be a population RATE
    (e.g. "flag the riskiest 5%"), not a probability -- that must NOT be
    used directly as a 0-1 decision_threshold."""
    artifact = {
        "model": _FakeModel(),
        "feature_cols": REQUIRED_FEATURES,
        "operating_point": {"mode": "rate", "value": 0.05},
    }

    adapter = ChurnModelAdapter(artifact)

    assert adapter.decision_threshold == ChurnModelAdapter.DEFAULT_DECISION_THRESHOLD


def test_decision_threshold_uses_score_based_operating_point():
    """A score-mode operating_point IS a real probability cutoff and
    should be used directly, unlike the rate-mode case above."""
    artifact = {
        "model": _FakeModel(),
        "feature_cols": REQUIRED_FEATURES,
        "operating_point": {"mode": "score", "value": 0.31},
    }

    adapter = ChurnModelAdapter(artifact)

    assert adapter.decision_threshold == 0.31


def test_decision_threshold_defaults_for_bare_model():
    adapter = ChurnModelAdapter(_FakeModel())

    assert adapter.decision_threshold == ChurnModelAdapter.DEFAULT_DECISION_THRESHOLD