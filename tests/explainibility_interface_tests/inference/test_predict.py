"""
test_predict.py

Covers: model loading (incl. missing artifact), feature validation (incl.
missing required feature), single-customer and batch prediction, top-N
reason code selection, reason-code mapping, and the full ChurnPredictor
pipeline as a black box (determinism, JSON-safety, output shape) — the
way FastAPI / the segmentation module actually calls it.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.ml.explainibility_inference.config.reason_code_lookup import load_reason_codes, select_top_reason_codes
from app.ml.explainibility_inference.inference.feature_contract import (
    FeatureValidationError,
    REQUIRED_FEATURES,
    validate_features,
)
from app.ml.explainibility_inference.inference.model_loader import ModelArtifactNotFoundError, load_model
from app.ml.explainibility_inference.inference.predict import DEFAULT_MODEL_PATH, DEFAULT_REASON_CODES_PATH, MODEL_VERSION


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def test_model_loading_succeeds():
    artifact = load_model(DEFAULT_MODEL_PATH)
    assert artifact is not None
    # Real artifacts are a dict — two shapes are valid, see
    # model_adapter.py's module docstring: an uncalibrated training
    # artifact ({"model": ..., "feature_cols": ..., "positive_class": ...})
    # or the final calibrated one ({"base_model": ..., "calibrator": ...,
    # "method": ..., "threshold": ..., "feature_cols": ...}). Confirm it's
    # one of those two recognized shapes rather than assuming either one
    # specifically — DEFAULT_MODEL_PATH may point at either, depending on
    # whether calibration.py has been run yet.
    assert isinstance(artifact, dict)
    assert ("model" in artifact) or ("base_model" in artifact), (
        f"Artifact dict has neither 'model' nor 'base_model' key — "
        f"unrecognized shape: {sorted(artifact.keys())}"
    )
    working_model = artifact["model"] if "model" in artifact else artifact["base_model"]
    assert hasattr(working_model, "predict_proba")


def test_model_loading_missing_artifact_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.joblib"
    with pytest.raises(ModelArtifactNotFoundError):
        load_model(missing_path)


# ---------------------------------------------------------------------------
# Feature validation
# ---------------------------------------------------------------------------


def test_feature_validation_passes_for_valid_input(sample_customer_df):
    validated = validate_features(sample_customer_df)
    assert "customer_unique_id" in validated.columns
    assert len(validated) == 1


def test_feature_validation_missing_required_feature_raises(sample_customer_df):
    broken = sample_customer_df.drop(columns=["monetary_value"])
    with pytest.raises(FeatureValidationError):
        validate_features(broken)


def test_feature_validation_null_value_raises(sample_customer_df):
    broken = sample_customer_df.copy()
    broken.loc[0, "avg_review_score"] = None
    with pytest.raises(FeatureValidationError):
        validate_features(broken)


def test_feature_validation_empty_dataframe_raises():
    with pytest.raises(FeatureValidationError):
        validate_features(pd.DataFrame())


def test_feature_validation_missing_id_column_raises(sample_customer_df):
    broken = sample_customer_df.drop(columns=["customer_unique_id"])
    with pytest.raises(FeatureValidationError):
        validate_features(broken)


def test_feature_validation_reorders_columns(sample_customer_df):
    shuffled = sample_customer_df[
        ["customer_unique_id"] + list(reversed(sample_customer_df.columns[1:]))
    ]
    validated = validate_features(shuffled)
    assert list(validated.columns) == ["customer_unique_id"] + REQUIRED_FEATURES


# ---------------------------------------------------------------------------
# Reason code mapping
# ---------------------------------------------------------------------------


def test_reason_codes_load_successfully():
    config = load_reason_codes(DEFAULT_REASON_CODES_PATH)
    assert "RC02" in config
    assert config["RC02"]["feature"] == "has_bad_review"


def test_top_n_reason_code_selection():
    config = load_reason_codes(DEFAULT_REASON_CODES_PATH)
    shap_values = {
        "has_bad_review": 0.9,  # positive -> RC02
        "avg_payment_installments": -0.5,  # unmapped feature (any direction) -> not selected
        "avg_delivery_days": 0.01,  # small magnitude, should not be selected
        "monetary_value": 0.02,  # not mapped to any reason code
        "avg_review_score": 0.0,
        "has_review_comment": 0.0,
        "avg_delivery_delay_days": 0.0,
        "is_delayed_delivery": 0.0,
    }

    codes = select_top_reason_codes(shap_values, config, top_n=1)

    assert codes == ["RC02"]


def test_reason_code_direction_filter_matches_predicted_outcome():
    """A large risk-LOWERING driver must not be selected as the reason
    for a customer the model predicts WILL churn -- predicted_positive
    restricts candidates to same-sign (risk-increasing) SHAP features."""
    config = load_reason_codes(DEFAULT_REASON_CODES_PATH)
    shap_values = {
        "avg_delivery_days": -2.26,  # biggest |SHAP|, but risk-LOWERING (mapped: RC04B)
        "freight_ratio": 1.07,  # smaller |SHAP|, risk-INCREASING (mapped: RC03)
        "has_bad_review": -0.24,
    }

    # Without a direction filter, the biggest driver wins regardless of
    # sign -- even though it contradicts a predicted-churn outcome.
    unfiltered = select_top_reason_codes(shap_values, config, top_n=1)
    assert unfiltered == ["RC04B"]

    # With predicted_positive=True (model predicts this customer WILL
    # churn), only risk-increasing drivers are eligible -- freight_ratio,
    # not the larger-magnitude but contradictory avg_delivery_days.
    filtered = select_top_reason_codes(
        shap_values, config, top_n=1, predicted_positive=True
    )
    assert filtered == ["RC03"]


def test_reason_code_selection_skips_unmapped_features():
    config = load_reason_codes(DEFAULT_REASON_CODES_PATH)
    shap_values = {"tenure_days": 0.99}  # no reason code configured for this
    codes = select_top_reason_codes(shap_values, config, top_n=3)
    assert codes == []


# ---------------------------------------------------------------------------
# predict() interface — single customer and batch
# ---------------------------------------------------------------------------


def test_predict_single_customer(churn_predictor, sample_customer_df):
    results = churn_predictor.predict(sample_customer_df)

    assert len(results) == 1
    record = results[0]
    assert record["customer_unique_id"] == sample_customer_df["customer_unique_id"].iloc[0]
    assert 0.0 <= record["churn_probability"] <= 1.0
    assert isinstance(record["shap_values"], dict)
    assert isinstance(record["reason_codes"], list)
    assert record["model_version"] == MODEL_VERSION
    assert "scored_at" in record

    # Output shape contract: exact key set, SHAP covers exactly the
    # feature contract, and the whole record round-trips through JSON
    # (this is what FastAPI actually has to do with it).
    assert set(record.keys()) == {
        "customer_unique_id",
        "churn_probability",
        "shap_values",
        "reason_codes",
        "model_version",
        "scored_at",
    }
    assert set(record["shap_values"].keys()) == set(REQUIRED_FEATURES)
    json.dumps(record)


def test_predict_batch(churn_predictor, sample_batch_df):
    results = churn_predictor.predict(sample_batch_df)

    assert len(results) == len(sample_batch_df)
    # Order must match input order, not just same set of ids.
    ids_in_order = [r["customer_unique_id"] for r in results]
    assert ids_in_order == list(sample_batch_df["customer_unique_id"])

    for record in results:
        assert 0.0 <= record["churn_probability"] <= 1.0
        assert set(record["shap_values"].keys()) == set(REQUIRED_FEATURES)
        json.dumps(record)


def test_predict_raises_on_missing_feature(churn_predictor, sample_customer_df):
    broken = sample_customer_df.drop(columns=["monetary_value"])
    with pytest.raises(FeatureValidationError):
        churn_predictor.predict(broken)


def test_predict_is_deterministic_for_same_input(churn_predictor, sample_customer_df):
    """Same input should produce the same probability and SHAP values across
    repeated calls (scored_at will naturally differ, everything else should
    not)."""
    first = churn_predictor.predict(sample_customer_df)[0]
    second = churn_predictor.predict(sample_customer_df)[0]

    assert first["churn_probability"] == second["churn_probability"]
    assert first["shap_values"] == second["shap_values"]
    assert first["reason_codes"] == second["reason_codes"]


def test_predict_reason_codes_are_subset_of_known_codes(churn_predictor, sample_batch_df):
    known_codes = set(load_reason_codes(DEFAULT_REASON_CODES_PATH).keys())
    results = churn_predictor.predict(sample_batch_df)

    for record in results:
        assert set(record["reason_codes"]).issubset(known_codes)