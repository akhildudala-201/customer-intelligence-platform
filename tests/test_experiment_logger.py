"""Unit tests for centralized experiment logger."""

import pandas as pd
import pytest
from app.ml.experiment_logger import log_experiment, get_experiment_history


def test_log_experiment_creation(tmp_path):
    metrics = {
        "roc_auc": 0.9189,
        "pr_auc": 0.7614,
        "balanced_accuracy": 0.8403,
        "precision": 0.7911,
        "recall": 0.6794,
        "f1": 0.7310,
    }

    run_id = log_experiment(
        model_name="LightGBM",
        strategy="scale_pos_weight",
        metrics=metrics,
        features_count=10,
        threshold="rate=0.05",
        artifacts_path="outputs/models/model.joblib",
        log_dir=tmp_path,
    )

    assert run_id == 1

    csv_file = tmp_path / "experiment_log.csv"
    md_file = tmp_path / "experiment_log.md"
    assert csv_file.exists()
    assert md_file.exists()

    df = pd.read_csv(csv_file)
    assert len(df) == 1
    assert df.iloc[0]["model"] == "LightGBM"
    assert df.iloc[0]["roc_auc"] == 0.9189
    assert df.iloc[0]["pr_auc"] == 0.7614
    assert df.iloc[0]["f1_score"] == 0.7310


def test_log_experiment_auto_increment(tmp_path):
    metrics = {"roc_auc": 0.8, "pr_auc": 0.7, "balanced_accuracy": 0.75, "precision": 0.7, "recall": 0.7, "f1": 0.7}

    id1 = log_experiment("ModelA", "strategy1", metrics, 5, 0.5, log_dir=tmp_path)
    id2 = log_experiment("ModelB", "strategy2", metrics, 8, 0.6, log_dir=tmp_path)

    assert id1 == 1
    assert id2 == 2

    history = get_experiment_history(log_dir=tmp_path)
    assert len(history) == 2
    assert list(history["run_id"]) == [1, 2]
    assert list(history["model"]) == ["ModelA", "ModelB"]
