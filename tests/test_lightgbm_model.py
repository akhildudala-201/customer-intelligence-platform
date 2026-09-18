"""
tests/test_lightgbm_model.py

Unit tests for LightGBM churn classification pipeline, ChurnLightGBM wrapper,
leakage separation, operating point selection, and model persistence.
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "app" / "ml"
for candidate in (str(PROJECT_ROOT), str(ML_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from app.ml.train_lightgbm_model import (
    ChurnLightGBM,
    split_features_target,
    apply_operating_point,
    psi,
    make_estimator,
    save_model,
    hyperparameter_tuning,
    CONFIG,
    DEFAULT_BEST_PARAMS,
    LEAKY_COLS,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def synthetic_imbalanced_data():
    """Create a reproducible small synthetic imbalanced dataset for fast testing."""
    np.random.seed(42)
    n_samples = 300
    n_features = 5

    X_raw = np.random.randn(n_samples, n_features)
    # 10% positive class
    linear_comb = 1.5 * X_raw[:, 0] - 1.2 * X_raw[:, 1] + 0.3 * X_raw[:, 2] - 2.0
    prob = 1 / (1 + np.exp(-linear_comb))
    y_raw = (prob > 0.6).astype(int)

    feature_cols = [f"feature_{i}" for i in range(n_features)]
    df_X = pd.DataFrame(X_raw, columns=feature_cols)
    s_y = pd.Series(y_raw, name="churn_label")
    return df_X, s_y


# =====================================================================
# ChurnLightGBM Wrapper Tests
# =====================================================================

def test_churn_lightgbm_initialization():
    model = ChurnLightGBM(n_estimators=10, random_state=42)
    assert model.class_weight == "balanced"
    assert model.threshold == 0.5
    assert model.is_fitted_ is False
    assert model.feature_names_ == []


def test_churn_lightgbm_fit_and_predict_shapes(synthetic_imbalanced_data):
    X, y = synthetic_imbalanced_data
    model = ChurnLightGBM(n_estimators=15, random_state=42)
    model.fit(X, y)

    assert model.is_fitted_ is True
    assert model.feature_names_ == list(X.columns)

    probs = model.predict_proba(X)
    assert len(probs) == len(X)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_churn_lightgbm_threshold_behavior(synthetic_imbalanced_data):
    X, y = synthetic_imbalanced_data
    model = ChurnLightGBM(n_estimators=15, random_state=42)
    model.fit(X, y)

    preds_low = model.predict(X, threshold=0.15)
    preds_high = model.predict(X, threshold=0.85)

    assert preds_low.sum() >= preds_high.sum()


def test_churn_lightgbm_unfitted_raises_error(synthetic_imbalanced_data):
    X, _ = synthetic_imbalanced_data
    model = ChurnLightGBM(n_estimators=10)

    with pytest.raises(ValueError, match="Model is not fitted"):
        model.predict_proba(X)

    with pytest.raises(ValueError, match="Model is not fitted"):
        model.get_feature_importance()


def test_churn_lightgbm_single_class_training_error():
    X = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
    y = pd.Series(np.zeros(50, dtype=int))
    model = ChurnLightGBM(n_estimators=10)

    with pytest.raises(ValueError, match="Training data requires at least two classes"):
        model.fit(X, y)


def test_churn_lightgbm_tune_threshold(synthetic_imbalanced_data):
    X, y = synthetic_imbalanced_data
    model = ChurnLightGBM(n_estimators=15, random_state=42)
    model.fit(X, y)

    best_t, best_score = model.tune_threshold(X, y, metric="balanced_accuracy")
    assert 0.01 <= best_t <= 0.99
    assert best_score > 0.0
    assert model.threshold == best_t


def test_churn_lightgbm_evaluate(synthetic_imbalanced_data):
    X, y = synthetic_imbalanced_data
    model = ChurnLightGBM(n_estimators=15, random_state=42)
    model.fit(X, y)

    metrics = model.evaluate(X, y)
    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "confusion_matrix" in metrics


def test_churn_lightgbm_feature_importance(synthetic_imbalanced_data):
    X, y = synthetic_imbalanced_data
    model = ChurnLightGBM(n_estimators=20, random_state=42)
    model.fit(X, y)

    imp = model.get_feature_importance()
    assert isinstance(imp, pd.DataFrame)
    assert "feature" in imp.columns
    assert "importance" in imp.columns
    assert len(imp) == X.shape[1]
    # Verify sorted in descending order
    assert list(imp["importance"]) == sorted(list(imp["importance"]), reverse=True)


# =====================================================================
# Pipeline Helper Function Tests
# =====================================================================

def test_split_features_target_strips_leaky_columns():
    df = pd.DataFrame({
        "customer_unique_id": ["c1", "c2", "c3"],
        "churn_label": [0, 1, 0],
        "monetary_value": [100.0, 200.0, 150.0],
        "recency_days": [10, 20, 30],       # In LEAKY_COLS
        "tenure_days": [100, 200, 300],     # In LEAKY_COLS
        "frequency": [1, 2, 1],             # In LEAKY_COLS
        "freight_ratio": [0.15, 0.20, 0.10],
    })

    X, y = split_features_target(df)

    assert "churn_label" not in X.columns
    assert "customer_unique_id" not in X.columns
    for leaky in ("recency_days", "tenure_days", "frequency"):
        assert leaky not in X.columns

    assert "monetary_value" in X.columns
    assert "freight_ratio" in X.columns
    assert list(y) == [0, 1, 0]


def test_apply_operating_point_rate_mode():
    # 100 probabilities from 0.01 to 1.00
    probs = np.linspace(0.01, 1.0, 100)

    # Flag top 5%
    preds = apply_operating_point(probs, mode="rate", value=0.05)
    assert preds.sum() == 5
    # The top 5 highest probabilities should be 1
    assert (preds[95:] == 1).all()
    assert (preds[:95] == 0).all()


def test_apply_operating_point_score_mode():
    probs = np.array([0.1, 0.45, 0.50, 0.75, 0.90])
    preds = apply_operating_point(probs, mode="score", value=0.50)
    assert list(preds) == [0, 0, 1, 1, 1]


def test_psi_identical_distribution():
    np.random.seed(42)
    dist = np.random.normal(10, 2, 500)
    # Identical distributions should have near-zero PSI (< 0.05)
    score = psi(dist, dist)
    assert score < 0.05


def test_psi_shifted_distribution():
    np.random.seed(42)
    dist1 = np.random.normal(0, 1, 500)
    dist2 = np.random.normal(5, 1, 500)  # heavily shifted
    score = psi(dist1, dist2)
    assert score > 0.25


def test_make_estimator_constructs_lgbm_with_weights():
    y_dummy = pd.Series([0] * 90 + [1] * 10)  # 9:1 imbalance
    estimator = make_estimator({"num_leaves": 16, "learning_rate": 0.05}, y_dummy)
    assert hasattr(estimator, "fit")
    assert estimator.n_estimators == 400


def test_hyperparameter_tuning_cached_fast_mode(synthetic_imbalanced_data):
    X, y = synthetic_imbalanced_data
    # Fast mode with TUNE_HYPERPARAMETERS = False
    params = hyperparameter_tuning(X, y)
    assert params == DEFAULT_BEST_PARAMS
    assert "num_leaves" in params
    assert "learning_rate" in params


def test_save_model_bundle_and_reload(tmp_path, synthetic_imbalanced_data):
    X, y = synthetic_imbalanced_data
    model = ChurnLightGBM(n_estimators=10, random_state=42)
    model.fit(X, y)

    results = {
        "TEST": {
            "roc_auc": 0.90,
            "pr_auc": 0.85,
            "precision": 0.80,
            "recall": 0.75,
            "f1": 0.77,
        }
    }

    # Temporarily override output dir to tmp_path
    original_out = CONFIG["OUTPUT_DIR"]
    CONFIG["OUTPUT_DIR"] = str(tmp_path) + "/"
    try:
        model_path, meta_path = save_model(
            model=model.model,
            feature_cols=list(X.columns),
            results=results,
            mode="rate",
            value=0.05,
            positive_class=1,
            best_params={"num_leaves": 23},
        )

        assert Path(model_path).exists()
        assert Path(meta_path).exists()

        # Reload model bundle and check predictions
        bundle = joblib.load(model_path)
        assert "model" in bundle
        assert "feature_cols" in bundle
        assert bundle["feature_cols"] == list(X.columns)
        assert bundle["operating_point"] == {"mode": "rate", "value": 0.05}

        reloaded_model = bundle["model"]
        reloaded_probs = reloaded_model.predict_proba(X)[:, 1]
        original_probs = model.predict_proba(X)
        np.testing.assert_allclose(original_probs, reloaded_probs, rtol=1e-5)

        # Check metadata JSON
        with open(meta_path) as fh:
            meta = json.load(fh)
        assert meta["operating_point"]["value"] == 0.05
        assert meta["metrics"]["TEST"]["roc_auc"] == 0.90
    finally:
        CONFIG["OUTPUT_DIR"] = original_out
