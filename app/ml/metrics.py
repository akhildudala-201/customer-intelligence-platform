"""Evaluation metrics calculation, threshold optimization, and reporting."""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
)


def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    y_prob: Optional[Union[np.ndarray, pd.Series, list]] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute binary classification metrics, AUC scores, and confusion matrix.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Optional positive class probabilities.
        threshold: Decision threshold used for predictions.

    Returns:
        Dictionary of calculated evaluation metrics and confusion counts.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    f2 = float(fbeta_score(y_true_arr, y_pred_arr, beta=2, zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    cm_dict = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    metrics: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "confusion_matrix": cm_dict,
        "threshold": round(float(threshold), 4),
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob).astype(float)
        if len(np.unique(y_true_arr)) > 1:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true_arr, y_prob_arr)), 4)
            metrics["pr_auc"] = round(float(average_precision_score(y_true_arr, y_prob_arr)), 4)
        else:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None

        try:
            # Clip probabilities to avoid numerical issues in log_loss
            eps = 1e-15
            clipped_prob = np.clip(y_prob_arr, eps, 1 - eps)
            metrics["log_loss"] = round(float(log_loss(y_true_arr, clipped_prob)), 4)
        except Exception:
            metrics["log_loss"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["log_loss"] = None

    return metrics


def find_optimal_threshold(
    y_true: Union[np.ndarray, pd.Series, list],
    y_prob: Union[np.ndarray, pd.Series, list],
    metric: str = "f1",
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Search for the decision threshold that maximizes a target metric.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted positive class probabilities.
        metric: Target metric to maximize ('f1', 'balanced_accuracy', etc.).
        thresholds: Array of thresholds to evaluate. Defaults to 0.01-0.99.

    Returns:
        Tuple of (optimal_threshold, best_score).
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_thresh = 0.5
    best_score = -1.0

    metric_funcs = {
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "f2": lambda yt, yp: fbeta_score(yt, yp, beta=2, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score,
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "accuracy": accuracy_score,
    }

    if metric not in metric_funcs:
        raise ValueError(f"Unsupported optimization metric '{metric}'. Valid: {list(metric_funcs.keys())}")

    fn = metric_funcs[metric]

    for t in thresholds:
        preds = (y_prob_arr >= t).astype(int)
        score = float(fn(y_true_arr, preds))
        if score > best_score:
            best_score = score
            best_thresh = float(t)

    return round(best_thresh, 4), round(best_score, 4)


def format_metrics_summary(metrics: Dict[str, Any], title: str = "Evaluation Metrics") -> str:
    """Format metrics dictionary into a readable tabular summary string."""
    cm = metrics.get("confusion_matrix", {})
    tn, fp, fn, tp = cm.get("tn", 0), cm.get("fp", 0), cm.get("fn", 0), cm.get("tp", 0)
    total = tn + fp + fn + tp

    lines = [
        f"=================== {title} ===================",
        f"  Decision Threshold:    {metrics.get('threshold')}",
        f"  Accuracy:              {metrics.get('accuracy')}",
        f"  Balanced Accuracy:     {metrics.get('balanced_accuracy')}",
        f"  Precision (Churn=1):   {metrics.get('precision')}",
        f"  Recall (Churn=1):      {metrics.get('recall')}",
        f"  F1-Score:              {metrics.get('f1')}",
        f"  F2-Score (Recall-wt):  {metrics.get('f2')}",
        f"  ROC-AUC:               {metrics.get('roc_auc')}",
        f"  PR-AUC (Avg Prec):     {metrics.get('pr_auc')}",
        f"  Log Loss:              {metrics.get('log_loss')}",
        "--------------------------------------------------",
        "  Confusion Matrix:",
        f"    True Negatives  (TN): {tn:>6} ({tn/total*100:5.1f}%) | False Positives (FP): {fp:>6} ({fp/total*100:5.1f}%)",
        f"    False Negatives (FN): {fn:>6} ({fn/total*100:5.1f}%) | True Positives  (TP): {tp:>6} ({tp/total*100:5.1f}%)",
        "==================================================",
    ]
    return "\n".join(lines)
