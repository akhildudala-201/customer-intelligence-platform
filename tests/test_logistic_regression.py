"""
tests/test_logistic_regression.py

Unit tests for Logistic Regression churn model, imbalance handling,
metrics computation, threshold tuning, and model persistence.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "app" / "ml"
sys.path.insert(0, str(ML_DIR))

from app.ml.logistic_regression import ChurnLogisticRegression, train_and_evaluate, separate_features_and_label
from app.ml.metrics import calculate_metrics, find_optimal_threshold, format_metrics_summary


# =====================================================================
# Metrics Tests
# =====================================================================

def test_calculate_metrics_perfect_prediction():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]

    metrics = calculate_metrics(y_true, y_pred, y_prob, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_calculate_metrics_known_confusion_matrix():
    # 2 TN, 1 FP, 1 FN, 2 TP
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1]

    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["confusion_matrix"]["tn"] == 2
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["fn"] == 1
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert metrics["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["recall"] == pytest.approx(2 / 3, abs=1e-4)


def test_calculate_metrics_single_class_edge_case():
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 1]
    y_prob = [0.1, 0.2, 0.3, 0.6]

    # Should cleanly capture UserWarning and not raise unhandled exception
    with pytest.warns(UserWarning):
        metrics = calculate_metrics(y_true, y_pred, y_prob)
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["accuracy"] == 0.75



def test_find_optimal_threshold():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Predicted probabilities where standard 0.5 threshold might not be optimal
    y_prob = np.array([0.1, 0.15, 0.25, 0.35, 0.38, 0.42, 0.75, 0.85])

    # At threshold 0.5, only 2 positive predictions (recall = 0.5)
    # At threshold ~0.37, all positives captured with 0 false positives!
    best_thresh, best_f1 = find_optimal_threshold(y_true, y_prob, metric="f1")

    assert 0.35 <= best_thresh <= 0.38
    assert best_f1 == 1.0


def test_format_metrics_summary_contains_key_metrics():
    metrics = {
        "threshold": 0.5,
        "accuracy": 0.85,
        "balanced_accuracy": 0.82,
        "precision": 0.78,
        "recall": 0.75,
        "f1": 0.76,
        "f2": 0.75,
        "roc_auc": 0.89,
        "pr_auc": 0.84,
        "log_loss": 0.35,
        "confusion_matrix": {"tn": 100, "fp": 15, "fn": 20, "tp": 65},
    }
    summary = format_metrics_summary(metrics, title="Test Set Results")
    assert "Test Set Results" in summary
    assert "Accuracy:" in summary
    assert "Precision (Churn=1):" in summary
    assert "Recall (Churn=1):" in summary
    assert "True Negatives  (TN):" in summary


# =====================================================================
# Model Architecture & Imbalance Handling Tests
# =====================================================================

@pytest.fixture
def synthetic_imbalanced_data():
    """
    Creates synthetic imbalanced churn dataset (e.g. 85% non-churn, 15% churn).
    """
    np.random.seed(42)
    n_samples = 400
    n_features = 5

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Imbalanced labels: only 15% churn
    linear_comb = 1.2 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] - 1.8
    prob = 1 / (1 + np.exp(-linear_comb))
    y = (prob > 0.6).astype(int)

    feature_names = [f"feat_{i}" for i in range(n_features)]
    df_X = pd.DataFrame(X, columns=feature_names)
    s_y = pd.Series(y, name="churn_label")

    return df_X, s_y


def test_model_initialization_defaults():
    model = ChurnLogisticRegression()
    assert model.class_weight == "balanced"
    assert model.threshold == 0.5
    assert model.is_fitted_ is False
    assert model.feature_names_ == []


def test_model_fit_and_predict_shapes(synthetic_imbalanced_data):
    X, y = synthetic_imbalanced_data
    model = ChurnLogisticRegression()
    model.fit(X, y)

    assert model.is_fitted_ is True
    assert model.feature_names_ == list(X.columns)

    probs = model.predict_proba(X)
    assert len(probs) == len(X)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_custom_threshold_behavior(synthetic_imbalanced_data):
    X, y = synthetic_imbalanced_data
    model = ChurnLogisticRegression()
    model.fit(X, y)

    # Lower threshold predicts more (or equal) positive cases
    preds_low = model.predict(X, threshold=0.2)
    preds_high = model.predict(X, threshold=0.8)

    assert preds_low.sum() >= preds_high.sum()


def test_imbalance_handling_class_weight_effect():
    """
    Verify that class_weight='balanced' gives higher recall on minority class
    compared to unweighted model on heavily skewed data.
    """
    np.random.seed(42)
    n = 500
    # 90% class 0, 10% class 1
    X = np.random.randn(n, 4)
    y = np.zeros(n, dtype=int)
    y[:50] = 1  # minority class
    X[:50, 0] += 1.0  # signal on feature 0

    df_X = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    s_y = pd.Series(y)

    unweighted = ChurnLogisticRegression(class_weight=None, random_state=42)
    unweighted.fit(df_X, s_y)
    rec_unweighted = unweighted.evaluate(df_X, s_y)["recall"]

    balanced = ChurnLogisticRegression(class_weight="balanced", random_state=42)
    balanced.fit(df_X, s_y)
    rec_balanced = balanced.evaluate(df_X, s_y)["recall"]

    # Balanced weighting must achieve higher or equal minority class recall
    assert rec_balanced >= rec_unweighted


def test_tune_threshold_updates_model(synthetic_imbalanced_data):
    X, y = synthetic_imbalanced_data
    split_idx = int(len(X) * 0.7)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]

    model = ChurnLogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)

    assert model.threshold == 0.5
    best_t, best_score = model.tune_threshold(X_val, y_val, metric="f1")

    assert model.threshold == best_t
    assert 0.01 <= best_t <= 0.99
    assert best_score >= 0.0


def test_get_feature_importance(synthetic_imbalanced_data):
    X, y = synthetic_imbalanced_data
    model = ChurnLogisticRegression()
    model.fit(X, y)

    fi = model.get_feature_importance()

    assert list(fi.columns) == ["feature", "coefficient", "odds_ratio", "abs_coefficient"]
    assert len(fi) == X.shape[1]
    # Check that it is sorted descending by absolute coefficient
    assert fi["abs_coefficient"].is_monotonic_decreasing
    # Verify odds ratio is exp(coefficient) within rounded precision
    np.testing.assert_allclose(fi["odds_ratio"], np.exp(fi["coefficient"]), atol=1e-3)


def test_unfitted_model_raises_errors(synthetic_imbalanced_data):
    X, y = synthetic_imbalanced_data
    model = ChurnLogisticRegression()

    with pytest.raises(ValueError, match="not fitted"):
        model.predict(X)

    with pytest.raises(ValueError, match="not fitted"):
        model.predict_proba(X)

    with pytest.raises(ValueError, match="not fitted"):
        model.get_feature_importance()

    with pytest.raises(ValueError, match="Cannot save"):
        model.save("dummy.joblib")


def test_model_persistence_save_and_load(tmp_path, synthetic_imbalanced_data):
    X, y = synthetic_imbalanced_data
    model = ChurnLogisticRegression(class_weight="balanced", threshold=0.42)
    model.fit(X, y)

    save_file = tmp_path / "model.joblib"
    saved_path = model.save(save_file)
    assert saved_path.exists()

    loaded_model = ChurnLogisticRegression.load(save_file)
    assert loaded_model.is_fitted_ is True
    assert loaded_model.threshold == 0.42
    assert loaded_model.class_weight == "balanced"
    assert loaded_model.feature_names_ == model.feature_names_

    # Verify identical predictions and probabilities
    np.testing.assert_array_equal(model.predict(X), loaded_model.predict(X))
    np.testing.assert_allclose(model.predict_proba(X), loaded_model.predict_proba(X))


# =====================================================================
# Pipeline and Separation Tests
# =====================================================================

def test_separate_features_and_label():
    df = pd.DataFrame({
        "customer_unique_id": ["c1", "c2"],
        "censored": [False, False],
        "first_purchase_date": ["2018-01-01", "2018-01-02"],
        "feature_a": [1.5, 2.5],
        "feature_b": [0.1, 0.9],
        "churn_label": [0, 1],
    })

    X, y = separate_features_and_label(df)

    assert list(X.columns) == ["feature_a", "feature_b"]
    assert list(y) == [0, 1]


def test_train_and_evaluate_workflow(tmp_path, synthetic_imbalanced_data):
    X, y = synthetic_imbalanced_data
    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train:n_train + n_val], y.iloc[n_train:n_train + n_val]
    X_test, y_test = X.iloc[n_train + n_val:], y.iloc[n_train + n_val:]

    save_path = tmp_path / "test_churn_model.joblib"

    model, results = train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        class_weight="balanced",
        tune_threshold=True,
        save_path=save_path,
        log_run=False,
    )

    assert model.is_fitted_ is True
    assert save_path.exists()
    assert "train" in results
    assert "val_tuned" in results
    assert "test" in results
    assert results["test"]["accuracy"] >= 0.0
    assert results["test"]["roc_auc"] is not None
    assert results["test"]["pr_auc"] is not None
