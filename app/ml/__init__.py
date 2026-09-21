"""Machine learning package exports for customer churn classification."""

from app.ml.experiment_logger import get_experiment_history, log_experiment
from app.ml.metrics import calculate_metrics, find_optimal_threshold, format_metrics_summary

__all__ = [
    "ChurnLogisticRegression",
    "ChurnLightGBM",
    "train_and_evaluate",
    "calculate_metrics",
    "find_optimal_threshold",
    "format_metrics_summary",
    "log_experiment",
    "get_experiment_history",
]


def __getattr__(name: str):
    """Load model implementations only when their optional dependencies are used."""
    if name in {"ChurnLogisticRegression", "train_and_evaluate"}:
        from app.ml.logistic_regression import (
            ChurnLogisticRegression,
            train_and_evaluate,
        )

        return {
            "ChurnLogisticRegression": ChurnLogisticRegression,
            "train_and_evaluate": train_and_evaluate,
        }[name]
    if name == "ChurnLightGBM":
        from app.ml.train_lightgbm_model import ChurnLightGBM

        return ChurnLightGBM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
