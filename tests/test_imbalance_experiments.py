"""
tests/test_imbalance_experiments.py

Unit tests for Person 3: Class Imbalance Module.
"""

import numpy as np
import pandas as pd
import pytest
from app.ml.imbalance_experiments import (
    analyze_class_imbalance,
    random_undersample,
    random_oversample,
    evaluate_strategy_predictions,
)


def test_analyze_class_imbalance():
    # 90 churned (1), 10 retained (0) -> 9:1 ratio
    y_train = np.array([0] * 10 + [1] * 90)
    y_val = np.array([0] * 5 + [1] * 45)

    analysis = analyze_class_imbalance(y_train, y_val=y_val)

    assert analysis["train_total"] == 100
    assert analysis["train_retained_count"] == 10
    assert analysis["train_retained_pct"] == 10.0
    assert analysis["train_churn_count"] == 90
    assert analysis["train_churn_pct"] == 90.0
    assert analysis["imbalance_ratio"] == 9.0
    assert analysis["val_total"] == 50
    assert analysis["val_churn_pct"] == 90.0
    assert analysis["theoretical_weights"][0] == 5.0  # 100 / (2 * 10)
    assert analysis["theoretical_weights"][1] == round(100 / (2 * 90), 4)


def test_random_undersample():
    X = pd.DataFrame({"feat": range(100)})
    # 20 minority (0), 80 majority (1)
    y = pd.Series([0] * 20 + [1] * 80)

    # 1:1 undersampling -> 20 minority, 20 majority = 40 total
    X_sub, y_sub = random_undersample(X, y, majority_to_minority_ratio=1.0, random_state=42)

    assert len(X_sub) == 40
    assert len(y_sub) == 40
    assert (y_sub == 0).sum() == 20
    assert (y_sub == 1).sum() == 20


def test_random_oversample():
    X = pd.DataFrame({"feat": range(100)})
    # 10 minority (0), 90 majority (1)
    y = pd.Series([0] * 10 + [1] * 90)

    # 1:1 oversampling -> 90 majority, 90 minority = 180 total
    X_over, y_over = random_oversample(X, y, majority_to_minority_ratio=1.0, random_state=42)

    assert len(X_over) == 180
    assert len(y_over) == 180
    assert (y_over == 0).sum() == 90
    assert (y_over == 1).sum() == 90


def test_evaluate_strategy_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.6, 0.4, 0.8])

    metrics = evaluate_strategy_predictions(y_true, y_pred, y_prob, threshold=0.5)

    assert metrics["threshold"] == 0.5
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["retained_recall_tn_rate"] == 0.5
    assert metrics["churn_recall_tp_rate"] == 0.5
    assert metrics["roc_auc"] is not None


def test_churn_lightgbm():
    from app.ml.train_lightgbm_model import ChurnLightGBM

    np.random.seed(42)
    X = pd.DataFrame({
        "f1": np.random.randn(100),
        "f2": np.random.randn(100),
    })
    y = pd.Series([0] * 15 + [1] * 85)

    model = ChurnLightGBM(class_weight="balanced", n_estimators=10, random_state=42)
    model.fit(X, y)

    assert model.is_fitted_
    probs = model.predict_proba(X)
    assert len(probs) == 100
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    preds = model.predict(X, threshold=0.5)
    assert len(preds) == 100
    assert set(np.unique(preds)).issubset({0, 1})

    best_t, best_score = model.tune_threshold(X, y, metric="balanced_accuracy")
    assert 0.0 < best_t < 1.0
    assert best_score >= 0.0

    metrics = model.evaluate(X, y)
    assert "balanced_accuracy" in metrics
    assert "confusion_matrix" in metrics

    importances = model.get_feature_importance()
    assert len(importances) == 2
    assert "feature" in importances.columns
    assert "importance" in importances.columns


def test_run_imbalance_experiments_lightgbm():
    from app.ml.imbalance_experiments import run_imbalance_experiments, run_lightgbm_imbalance_experiments

    np.random.seed(42)
    X_train = pd.DataFrame({"f1": np.random.randn(60), "f2": np.random.randn(60)})
    y_train = pd.Series([0] * 10 + [1] * 50)

    X_val = pd.DataFrame({"f1": np.random.randn(30), "f2": np.random.randn(30)})
    y_val = pd.Series([0] * 5 + [1] * 25)

    X_test = pd.DataFrame({"f1": np.random.randn(30), "f2": np.random.randn(30)})
    y_test = pd.Series([0] * 5 + [1] * 25)

    results_df, trained_models = run_imbalance_experiments(
        X_train, y_train, X_val, y_val, X_test, y_test, model_type="lightgbm"
    )

    assert len(results_df) == 8
    assert "Strategy" in results_df.columns
    assert "Balanced Acc" in results_df.columns
    assert "1. Unweighted Baseline" in results_df["Strategy"].values
    assert len(trained_models) == 8

    # Test convenience wrapper
    df_wrap, _ = run_lightgbm_imbalance_experiments(X_train, y_train, X_val, y_val, X_test, y_test)
    assert len(df_wrap) == 8

