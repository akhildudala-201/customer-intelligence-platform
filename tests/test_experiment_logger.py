"""Unit tests for centralized experiment logger."""

from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest

from app.ml.experiment_logger import (
    COLUMNS,
    _format_markdown_table,
    get_experiment_history,
    log_experiment,
)


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
    metrics = {
        "roc_auc": 0.8,
        "pr_auc": 0.7,
        "balanced_accuracy": 0.75,
        "precision": 0.7,
        "recall": 0.7,
        "f1": 0.7,
    }

    id1 = log_experiment("ModelA", "strategy1", metrics, 5, 0.5, log_dir=tmp_path)
    id2 = log_experiment("ModelB", "strategy2", metrics, 8, 0.6, log_dir=tmp_path)

    assert id1 == 1
    assert id2 == 2

    history = get_experiment_history(log_dir=tmp_path)
    assert len(history) == 2
    assert list(history["run_id"]) == [1, 2]
    assert list(history["model"]) == ["ModelA", "ModelB"]


def test_get_experiment_history_nonexistent(tmp_path):
    """Verify get_experiment_history returns empty DataFrame when log file is absent."""
    empty_dir = tmp_path / "nonexistent_dir"
    history = get_experiment_history(log_dir=empty_dir)

    assert isinstance(history, pd.DataFrame)
    assert history.empty
    assert list(history.columns) == COLUMNS


def test_log_experiment_corrupted_csv_handling(tmp_path):
    """Verify graceful recovery and reset when experiment_log.csv fails to read."""
    csv_file = tmp_path / "experiment_log.csv"
    csv_file.write_text("invalid,csv,syntax\nthat,cannot,be,parsed\n", encoding="utf-8")

    with patch("pandas.read_csv", side_effect=Exception("Corrupted CSV file")):
        run_id = log_experiment(
            model_name="FallbackModel",
            strategy="recovery_test",
            metrics={"roc_auc": 0.85},
            features_count=14,
            threshold=0.5,
            log_dir=tmp_path,
        )

    assert run_id == 1
    recovered_history = get_experiment_history(log_dir=tmp_path)
    assert len(recovered_history) == 1
    assert recovered_history.iloc[0]["model"] == "FallbackModel"


def test_log_experiment_partial_and_none_metrics(tmp_path):
    """Verify logging when metrics dictionary is sparse or contains None values."""
    metrics = {
        "roc_auc": 0.82,
        "f1_score": 0.75,  # uses f1_score instead of f1
        # pr_auc, balanced_accuracy, precision, recall omitted
    }

    run_id = log_experiment(
        model_name="SparseModel",
        strategy="partial_metrics",
        metrics=metrics,
        features_count=14,
        threshold="0.59",
        log_dir=tmp_path,
    )

    assert run_id == 1
    history = get_experiment_history(log_dir=tmp_path)
    assert history.iloc[0]["roc_auc"] == 0.82
    assert history.iloc[0]["f1_score"] == 0.75
    assert pd.isna(history.iloc[0]["pr_auc"])
    assert pd.isna(history.iloc[0]["precision"])


def test_format_markdown_table():
    """Verify markdown table formatter handles DataFrames with null values properly."""
    sample_df = pd.DataFrame(
        [
            {"run_id": 1, "model": "ModelA", "roc_auc": 0.88, "precision": None},
            {"run_id": 2, "model": "ModelB", "roc_auc": None, "precision": 0.95},
        ]
    )
    md_output = _format_markdown_table(sample_df)

    assert "| run_id | model | roc_auc | precision |" in md_output
    assert "| --- | --- | --- | --- |" in md_output
    assert "| 1 | ModelA | 0.88 |  |" in md_output
    assert "| 2 | ModelB |  | 0.95 |" in md_output


def test_log_experiment_default_dir(tmp_path, monkeypatch):
    """Verify fallback to DEFAULT_LOG_DIR when log_dir is omitted."""
    monkeypatch.setattr("app.ml.experiment_logger.DEFAULT_LOG_DIR", tmp_path)

    run_id = log_experiment(
        model_name="DefaultDirModel",
        strategy="default_dir_test",
        metrics={"roc_auc": 0.90},
        features_count=14,
        threshold=0.5,
        log_dir=None,
    )

    assert run_id == 1
    assert (tmp_path / "experiment_log.csv").exists()
    assert (tmp_path / "experiment_log.md").exists()

    history = get_experiment_history(log_dir=None)
    assert len(history) == 1
    assert history.iloc[0]["model"] == "DefaultDirModel"
