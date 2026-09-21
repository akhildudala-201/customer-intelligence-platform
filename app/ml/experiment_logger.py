"""Centralized experiment logger for tracking ML model runs and evaluation metrics."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs" / "reports"
DEFAULT_CSV_PATH = DEFAULT_LOG_DIR / "experiment_log.csv"
DEFAULT_MD_PATH = DEFAULT_LOG_DIR / "experiment_log.md"

COLUMNS = [
    "run_id",
    "timestamp",
    "model",
    "strategy",
    "features_count",
    "threshold",
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1_score",
]



def _format_markdown_table(df: pd.DataFrame) -> str:
    """Convert experiment log DataFrame into a GitHub markdown table."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(v) if pd.notnull(v) else "" for v in r) + " |" for r in df.values]
    return "\n".join([header, sep] + rows)


def log_experiment(
    model_name: str,
    strategy: str,
    metrics: Dict[str, Any],
    features_count: int,
    threshold: Union[float, str] = 0.5,
    artifacts_path: Optional[str] = None,
    log_dir: Optional[Union[str, Path]] = None,
) -> int:
    
    target_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_file = target_dir / "experiment_log.csv"
    md_file = target_dir / "experiment_log.md"

    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            next_id = int(df["run_id"].max() + 1) if not df.empty and "run_id" in df else 1
        except Exception:
            df = pd.DataFrame(columns=COLUMNS)
            next_id = 1
    else:
        df = pd.DataFrame(columns=COLUMNS)
        next_id = 1

    # Extract metrics safely supporting various naming conventions
    roc_auc = metrics.get("roc_auc")
    pr_auc = metrics.get("pr_auc")
    bal_acc = metrics.get("balanced_accuracy")
    prec = metrics.get("precision")
    rec = metrics.get("recall")
    f1 = metrics.get("f1") or metrics.get("f1_score")

    entry = {
        "run_id": next_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "strategy": strategy,
        "features_count": int(features_count),
        "threshold": str(threshold),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "pr_auc": round(float(pr_auc), 4) if pr_auc is not None else None,
        "balanced_accuracy": round(float(bal_acc), 4) if bal_acc is not None else None,
        "precision": round(float(prec), 4) if prec is not None else None,
        "recall": round(float(rec), 4) if rec is not None else None,
        "f1_score": round(float(f1), 4) if f1 is not None else None,
    }

    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(csv_file, index=False)

    # Write Markdown summary
    with md_file.open("w", encoding="utf-8") as f:
        f.write("# Centralized Machine Learning Experiment Log\n\n")
        f.write("Historical record of all trained churn models, weighting strategies, and test set metrics.\n\n")
        f.write(_format_markdown_table(df))
        f.write("\n")

    return next_id


def get_experiment_history(log_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Retrieve full experiment history as a pandas DataFrame."""
    target_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    csv_file = target_dir / "experiment_log.csv"
    if not csv_file.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(csv_file)
