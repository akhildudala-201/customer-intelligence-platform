import numpy as np
import pandas as pd
import pytest

from pathlib import Path


from app.ml.model_comparison import (
    LoadedLightGBMModel,
    tune_and_evaluate,
    build_comparison_table,
    summarize_winner,
    export_probabilities,
)

from app.ml.threshold_analysis import (
    threshold_sweep,
    best_threshold_per_metric,
    compare_best_thresholds,
)



class FakeModel:
    """
    Simple fake model used for testing.

    It behaves like a trained classification model but does
    not require loading a real Logistic Regression or LightGBM
    model.
    """

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)
        self.threshold = 0.5
        self.metrics_ = {}

    def predict_proba(self, X):
        return self.probabilities

    def tune_threshold(self, X_val, y_val, metric="f1", thresholds=None):
        """
        Simple deterministic threshold tuning for testing.
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)

        best_threshold = None
        best_score = -1

        for threshold in thresholds:
            predictions = (self.probabilities >= threshold).astype(int)

            tp = np.sum((predictions == 1) & (y_val == 1))
            fp = np.sum((predictions == 1) & (y_val == 0))
            fn = np.sum((predictions == 0) & (y_val == 1))
            tn = np.sum((predictions == 0) & (y_val == 0))

            if metric == "recall":
                score = tp / (tp + fn) if (tp + fn) > 0 else 0

            elif metric == "precision":
                score = tp / (tp + fp) if (tp + fp) > 0 else 0

            elif metric == "f1":
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                if precision + recall == 0:
                    score = 0
                else:
                    score = 2 * precision * recall / (precision + recall)

            elif metric == "balanced_accuracy":
                sensitivity = (
                    tp / (tp + fn)
                    if (tp + fn) > 0
                    else 0
                )

                specificity = (
                    tn / (tn + fp)
                    if (tn + fp) > 0
                    else 0
                )

                score = (sensitivity + specificity) / 2

            else:
                score = 0

            if score > best_score:
                best_score = score
                best_threshold = threshold

        self.threshold = best_threshold

        return best_threshold, best_score

    def evaluate(self, X, y, threshold=None):
        """
        Simple evaluation function used by tune_and_evaluate().
        """

        if threshold is None:
            threshold = self.threshold

        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int)

        tp = np.sum((predictions == 1) & (y == 1))
        fp = np.sum((predictions == 1) & (y == 0))
        fn = np.sum((predictions == 0) & (y == 1))
        tn = np.sum((predictions == 0) & (y == 0))

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        sensitivity = recall

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        )

        balanced_accuracy = (
            sensitivity + specificity
        ) / 2

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": balanced_accuracy,
        }

        self.metrics_ = metrics

        return metrics


# TEST DATA

@pytest.fixture
def sample_data():
    """
    Small dataset used by multiple tests.
    """

    X = pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5, 6],
        "feature_2": [10, 20, 30, 40, 50, 60],
    })

    y = pd.Series([0, 1, 1, 0, 1, 0])

    probabilities = np.array([
        0.10,
        0.80,
        0.90,
        0.20,
        0.70,
        0.30,
    ])

    return X, y, probabilities



# MODEL COMPARISON TESTS


def test_build_comparison_table():
    """
    Test that model comparison correctly determines
    which model has the better metric value.
    """

    logreg_result = {
        "test_metrics": {
            "roc_auc": 0.70,
            "pr_auc": 0.60,
            "balanced_accuracy": 0.65,
            "precision": 0.80,
            "recall": 0.70,
            "f1": 0.75,
            "f2": 0.72,
            "log_loss": 0.50,
        }
    }

    lgbm_result = {
        "test_metrics": {
            "roc_auc": 0.85,
            "pr_auc": 0.80,
            "balanced_accuracy": 0.78,
            "precision": 0.90,
            "recall": 0.75,
            "f1": 0.82,
            "f2": 0.80,
            "log_loss": 0.35,
        }
    }

    result = build_comparison_table(
        logreg_result,
        lgbm_result
    )

    assert isinstance(result, pd.DataFrame)

    assert len(result) == 8

    assert result.loc[
        result["metric"] == "roc_auc", "better"
    ].iloc[0] == "LightGBM"

    assert result.loc[
        result["metric"] == "precision", "better"
    ].iloc[0] == "LightGBM"

    assert result.loc[
        result["metric"] == "log_loss", "better"
    ].iloc[0] == "LightGBM"


def test_build_comparison_table_logreg_wins():
    """
    Test that Logistic Regression is selected when
    its metric value is higher.
    """

    logreg_result = {
        "test_metrics": {
            "roc_auc": 0.90,
            "pr_auc": 0.85,
            "balanced_accuracy": 0.80,
            "precision": 0.95,
            "recall": 0.90,
            "f1": 0.92,
            "f2": 0.91,
            "log_loss": 0.20,
        }
    }

    lgbm_result = {
        "test_metrics": {
            "roc_auc": 0.70,
            "pr_auc": 0.65,
            "balanced_accuracy": 0.60,
            "precision": 0.70,
            "recall": 0.65,
            "f1": 0.67,
            "f2": 0.66,
            "log_loss": 0.50,
        }
    }

    result = build_comparison_table(
        logreg_result,
        lgbm_result
    )

    assert all(
        result["better"] == "Logistic Regression"
    )


def test_build_comparison_table_tie():
    """
    Test that equal metric values produce 'tie'.
    """

    logreg_result = {
        "test_metrics": {
            "roc_auc": 0.80,
            "pr_auc": 0.70,
            "balanced_accuracy": 0.75,
            "precision": 0.80,
            "recall": 0.70,
            "f1": 0.75,
            "f2": 0.73,
            "log_loss": 0.40,
        }
    }

    lgbm_result = {
        "test_metrics": {
            "roc_auc": 0.80,
            "pr_auc": 0.70,
            "balanced_accuracy": 0.75,
            "precision": 0.80,
            "recall": 0.70,
            "f1": 0.75,
            "f2": 0.73,
            "log_loss": 0.40,
        }
    }

    result = build_comparison_table(
        logreg_result,
        lgbm_result
    )

    assert all(result["better"] == "tie")


def test_summarize_winner():
    """
    Test winner summary generation.
    """

    comparison_df = pd.DataFrame({
        "metric": [
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
        ],
        "logistic_regression": [
            0.70,
            0.60,
            0.75,
            0.70,
        ],
        "lightgbm": [
            0.80,
            0.75,
            0.85,
            0.65,
        ],
        "better": [
            "LightGBM",
            "LightGBM",
            "LightGBM",
            "Logistic Regression",
        ],
    })

    result = summarize_winner(comparison_df)

    assert isinstance(result, str)

    assert "LightGBM" in result
    assert "Logistic Regression" in result

    assert "3/4" in result
    assert "1/4" in result


def test_tune_and_evaluate(sample_data):
    """
    Test that tune_and_evaluate:

    1. Finds a threshold.
    2. Evaluates on validation data.
    3. Evaluates on test data.
    4. Returns all expected fields.
    """

    X, y, probabilities = sample_data

    model = FakeModel(probabilities)

    result = tune_and_evaluate(
        model=model,
        X_val=X,
        y_val=y,
        X_test=X,
        y_test=y,
        tune_metric="f1",
    )

    assert isinstance(result, dict)

    assert "threshold" in result
    assert "val_score_at_tuning" in result
    assert "val_metrics" in result
    assert "test_metrics" in result

    assert 0 <= result["threshold"] <= 1

    assert isinstance(
        result["val_metrics"],
        dict
    )

    assert isinstance(
        result["test_metrics"],
        dict
    )


def test_export_probabilities(tmp_path, sample_data):
    """
    Test that predicted probabilities are exported
    correctly to CSV.
    """

    X, y, probabilities = sample_data

    model = FakeModel(probabilities)

    output = export_probabilities(
        model=model,
        X=X,
        y=y,
        split_name="test",
        model_name="FakeModel",
        out_dir=tmp_path,
    )

    assert isinstance(output, pd.DataFrame)

    assert len(output) == len(X)

    assert "row_index" in output.columns
    assert "y_true" in output.columns
    assert "y_prob" in output.columns
    assert "model" in output.columns
    assert "split" in output.columns

    expected_file = (
        tmp_path /
        "probabilities_FakeModel_test.csv"
    )

    assert expected_file.exists()

    saved_df = pd.read_csv(expected_file)

    assert len(saved_df) == len(X)

    assert list(saved_df["y_prob"]) == list(
        probabilities
    )



# LIGHTGBM ADAPTER TESTS


class FakeRawLightGBM:
    """
    Fake LightGBM-like estimator.
    """

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, X):
        """
        Return probability for class 0 and class 1.
        """

        return np.column_stack([
            1 - self.probabilities,
            self.probabilities,
        ])


def test_lightgbm_probability_no_inversion():
    """
    If positive_class == 1, probability should not be inverted.
    """

    raw_model = FakeRawLightGBM(
        [0.2, 0.7, 0.9]
    )

    artifact = {
        "model": raw_model,
        "feature_cols": [
            "feature_1",
            "feature_2",
        ],
        "positive_class": 1,
    }

    model = LoadedLightGBMModel(artifact)

    X = pd.DataFrame({
        "feature_1": [1, 2, 3],
        "feature_2": [4, 5, 6],
    })

    result = model.predict_proba(X)

    expected = np.array([
        0.2,
        0.7,
        0.9,
    ])

    np.testing.assert_array_almost_equal(
        result,
        expected
    )


def test_lightgbm_probability_inversion():
    """
    If positive_class != 1, probability should be inverted.

    Example:
        raw probability = 0.2
        churn probability = 1 - 0.2 = 0.8
    """

    raw_model = FakeRawLightGBM(
        [0.2, 0.7, 0.9]
    )

    artifact = {
        "model": raw_model,
        "feature_cols": [
            "feature_1",
            "feature_2",
        ],
        "positive_class": 0,
    }

    model = LoadedLightGBMModel(artifact)

    X = pd.DataFrame({
        "feature_1": [1, 2, 3],
        "feature_2": [4, 5, 6],
    })

    result = model.predict_proba(X)

    expected = np.array([
        0.8,
        0.3,
        0.1,
    ])

    np.testing.assert_array_almost_equal(
        result,
        expected
    )


def test_lightgbm_missing_columns():
    """
    Test that missing feature columns raise ValueError.
    """

    raw_model = FakeRawLightGBM(
        [0.2, 0.7]
    )

    artifact = {
        "model": raw_model,
        "feature_cols": [
            "feature_1",
            "feature_2",
        ],
        "positive_class": 1,
    }

    model = LoadedLightGBMModel(artifact)

    X = pd.DataFrame({
        "feature_1": [1, 2],
    })

    with pytest.raises(ValueError):
        model.predict_proba(X)


def test_lightgbm_predict():
    """
    Test binary predictions using a threshold.
    """

    raw_model = FakeRawLightGBM(
        [0.2, 0.7, 0.8, 0.3]
    )

    artifact = {
        "model": raw_model,
        "feature_cols": [
            "feature_1",
        ],
        "positive_class": 1,
    }

    model = LoadedLightGBMModel(artifact)

    X = pd.DataFrame({
        "feature_1": [1, 2, 3, 4],
    })

    predictions = model.predict(
        X,
        threshold=0.5
    )

    expected = np.array([
        0,
        1,
        1,
        0,
    ])

    np.testing.assert_array_equal(
        predictions,
        expected
    )



# THRESHOLD ANALYSIS TESTS


def test_threshold_sweep(sample_data):
    """
    Test threshold_sweep().

    The function should:
    - create multiple threshold rows
    - calculate precision
    - calculate recall
    - calculate F1
    - calculate balanced accuracy
    """

    X, y, probabilities = sample_data

    model = FakeModel(probabilities)

    thresholds = np.array([
        0.2,
        0.5,
        0.8,
    ])

    result = threshold_sweep(
        model=model,
        X=X,
        y=y,
        thresholds=thresholds,
    )

    assert isinstance(result, pd.DataFrame)

    assert len(result) == 3

    expected_columns = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
    ]

    for column in expected_columns:
        assert column in result.columns

    assert list(result["threshold"]) == [
        0.2,
        0.5,
        0.8,
    ]


def test_threshold_sweep_default_thresholds(sample_data):
    """
    Test the default threshold grid.

    threshold_analysis.py uses:
        np.linspace(0.01, 0.99, 99)
    """

    X, y, probabilities = sample_data

    model = FakeModel(probabilities)

    result = threshold_sweep(
        model=model,
        X=X,
        y=y,
    )

    assert len(result) == 99

    assert result.iloc[0]["threshold"] == pytest.approx(
        0.01
    )

    assert result.iloc[-1]["threshold"] == pytest.approx(
        0.99
    )


def test_best_threshold_per_metric(sample_data):
    """
    Test finding the best threshold for each metric.
    """

    X, y, probabilities = sample_data

    model = FakeModel(probabilities)

    result = best_threshold_per_metric(
        model=model,
        X=X,
        y=y,
    )

    assert isinstance(result, pd.DataFrame)

    expected_metrics = [
        "f1",
        "balanced_accuracy",
        "precision",
        "recall",
    ]

    assert list(result["metric"]) == expected_metrics

    assert "best_threshold" in result.columns
    assert "best_score" in result.columns

    assert len(result) == 4

    assert all(
        (result["best_threshold"] >= 0) &
        (result["best_threshold"] <= 1)
    )


def test_compare_best_thresholds():
    """
    Test comparison of optimal thresholds between
    Logistic Regression and LightGBM.
    """

    logreg_best = pd.DataFrame({
        "metric": [
            "f1",
            "balanced_accuracy",
            "precision",
            "recall",
        ],
        "best_threshold": [
            0.40,
            0.50,
            0.60,
            0.30,
        ],
        "best_score": [
            0.70,
            0.75,
            0.80,
            0.65,
        ],
    })

    lgbm_best = pd.DataFrame({
        "metric": [
            "f1",
            "balanced_accuracy",
            "precision",
            "recall",
        ],
        "best_threshold": [
            0.55,
            0.65,
            0.70,
            0.45,
        ],
        "best_score": [
            0.82,
            0.78,
            0.85,
            0.60,
        ],
    })

    result = compare_best_thresholds(
        logreg_best,
        lgbm_best
    )

    assert isinstance(result, pd.DataFrame)

    assert len(result) == 4

    expected_columns = [
        "metric",
        "best_threshold_logreg",
        "best_score_logreg",
        "best_threshold_lgbm",
        "best_score_lgbm",
        "score_gap",
        "better",
    ]

    assert list(result.columns) == expected_columns

    f1_row = result[
        result["metric"] == "f1"
    ].iloc[0]

    assert f1_row["best_threshold_logreg"] == 0.40
    assert f1_row["best_threshold_lgbm"] == 0.55

    assert f1_row["better"] == "LightGBM"

    assert f1_row["score_gap"] == pytest.approx(
        0.12
    )


def test_compare_best_thresholds_logreg_wins():
    """
    Test the case where Logistic Regression has
    the higher score.
    """

    logreg_best = pd.DataFrame({
        "metric": ["f1"],
        "best_threshold": [0.40],
        "best_score": [0.90],
    })

    lgbm_best = pd.DataFrame({
        "metric": ["f1"],
        "best_threshold": [0.60],
        "best_score": [0.75],
    })

    result = compare_best_thresholds(
        logreg_best,
        lgbm_best
    )

    assert result.iloc[0]["better"] == (
        "Logistic Regression"
    )

    assert result.iloc[0]["score_gap"] == pytest.approx(
        -0.15
    )


def test_compare_best_thresholds_tie():
    """
    Test equal scores.
    """

    logreg_best = pd.DataFrame({
        "metric": ["f1"],
        "best_threshold": [0.50],
        "best_score": [0.80],
    })

    lgbm_best = pd.DataFrame({
        "metric": ["f1"],
        "best_threshold": [0.50],
        "best_score": [0.80],
    })

    result = compare_best_thresholds(
        logreg_best,
        lgbm_best
    )

    assert result.iloc[0]["better"] == "tie"

    assert result.iloc[0]["score_gap"] == 0


# ============================================================
# TEST SUMMARY
# ============================================================

def test_threshold_values_are_valid(sample_data):
    """
    Make sure all calculated threshold values remain
    between 0 and 1.
    """

    X, y, probabilities = sample_data

    model = FakeModel(probabilities)

    result = best_threshold_per_metric(
        model,
        X,
        y
    )

    assert all(
        result["best_threshold"].between(0, 1)
    )


def test_metrics_are_valid(sample_data):
    """
    Metric values should remain between 0 and 1.
    """

    X, y, probabilities = sample_data

    model = FakeModel(probabilities)

    result = threshold_sweep(
        model,
        X,
        y,
        thresholds=[0.2, 0.5, 0.8]
    )

    for metric in [
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
    ]:
        assert all(
            result[metric].between(0, 1)
        )
