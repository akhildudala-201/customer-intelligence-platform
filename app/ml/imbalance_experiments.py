"""Class imbalance analysis, resampling strategies, and model benchmarking."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[1]

for candidate in (str(PROJECT_ROOT), str(APP_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from app.ml.logistic_regression import (
    ChurnLogisticRegression,
    load_model_ready_data,
)
from app.ml.train_lightgbm_model import ChurnLightGBM
from app.ml.metrics import find_optimal_threshold


def analyze_class_imbalance(
    y_train: Union[pd.Series, np.ndarray],
    y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    y_test: Optional[Union[pd.Series, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Compute class distributions, imbalance ratios, and theoretical balanced weights."""
    y_tr = np.asarray(y_train).astype(int)
    n_total = len(y_tr)
    n_retained = int((y_tr == 0).sum())
    n_churned = int((y_tr == 1).sum())

    pct_retained = (n_retained / n_total) * 100.0
    pct_churned = (n_churned / n_total) * 100.0
    ratio = n_churned / max(n_retained, 1)

    # Balanced weight formula: N / (n_classes * n_samples_c)
    w_retained = n_total / (2.0 * n_retained)
    w_churned = n_total / (2.0 * n_churned)

    analysis = {
        "train_total": n_total,
        "train_retained_count": n_retained,
        "train_retained_pct": round(pct_retained, 2),
        "train_churn_count": n_churned,
        "train_churn_pct": round(pct_churned, 2),
        "imbalance_ratio": round(ratio, 2),
        "theoretical_weights": {
            0: round(float(w_retained), 4),
            1: round(float(w_churned), 4),
        },
        "effective_penalty_ratio": round(w_retained / w_churned, 2),
    }

    if y_val is not None:
        y_v = np.asarray(y_val).astype(int)
        analysis["val_total"] = len(y_v)
        analysis["val_churn_pct"] = round(float((y_v == 1).mean() * 100.0), 2)

    if y_test is not None:
        y_te = np.asarray(y_test).astype(int)
        analysis["test_total"] = len(y_te)
        analysis["test_churn_pct"] = round(float((y_te == 1).mean() * 100.0), 2)

    return analysis


def format_imbalance_report(analysis: Dict[str, Any]) -> str:
    """Format imbalance analysis into a text summary."""
    return f"""
=====================================================================
CLASS IMBALANCE ANALYSIS REPORT
=====================================================================
  Total Training Samples:       {analysis['train_total']:,}
  Retained Customers (Class 0): {analysis['train_retained_count']:,} ({analysis['train_retained_pct']}%)  <-- MINORITY
  Churned Customers  (Class 1): {analysis['train_churn_count']:,} ({analysis['train_churn_pct']}%)  <-- MAJORITY
  
  Imbalance Ratio:              {analysis['imbalance_ratio']}:1 (Majority to Minority)
  Theoretical 'Balanced' Weights:
    - Class 0 (Retained):       {analysis['theoretical_weights'][0]}x
    - Class 1 (Churn):          {analysis['theoretical_weights'][1]}x
  Effective Penalty Ratio:      {analysis['effective_penalty_ratio']}x on Minority Errors
---------------------------------------------------------------------
  Key Implication:
  A naive majority classifier achieves {analysis['train_churn_pct']}% accuracy by predicting
  everyone churns, with 0.0% recall on retained customers.
  Cost-sensitive weighting or threshold tuning is mandatory.
=====================================================================
"""


def random_undersample(
    X: pd.DataFrame,
    y: pd.Series,
    majority_to_minority_ratio: float = 1.0,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Undersample majority class to reach desired majority-to-minority ratio."""
    y_s = pd.Series(y).reset_index(drop=True)
    X_df = pd.DataFrame(X).reset_index(drop=True)

    idx_minority = y_s[y_s == 0].index
    idx_majority = y_s[y_s == 1].index

    n_minority = len(idx_minority)
    n_majority_desired = int(n_minority * majority_to_minority_ratio)

    rng = np.random.RandomState(random_state)
    chosen_majority = rng.choice(idx_majority, size=min(n_majority_desired, len(idx_majority)), replace=False)

    sampled_idx = np.concatenate([idx_minority, chosen_majority])
    rng.shuffle(sampled_idx)

    return X_df.iloc[sampled_idx].reset_index(drop=True), y_s.iloc[sampled_idx].reset_index(drop=True)


def random_oversample(
    X: pd.DataFrame,
    y: pd.Series,
    majority_to_minority_ratio: float = 1.0,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Oversample minority class with replacement to reach desired ratio."""
    y_s = pd.Series(y).reset_index(drop=True)
    X_df = pd.DataFrame(X).reset_index(drop=True)

    idx_minority = y_s[y_s == 0].index
    idx_majority = y_s[y_s == 1].index

    n_majority = len(idx_majority)
    n_minority_desired = int(n_majority / majority_to_minority_ratio)

    rng = np.random.RandomState(random_state)
    chosen_minority = rng.choice(idx_minority, size=n_minority_desired, replace=True)

    sampled_idx = np.concatenate([chosen_minority, idx_majority])
    rng.shuffle(sampled_idx)

    return X_df.iloc[sampled_idx].reset_index(drop=True), y_s.iloc[sampled_idx].reset_index(drop=True)


def evaluate_strategy_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Calculate classification metrics for both majority and minority classes."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    retained_recall = tn / max(tn + fp, 1)
    retained_precision = tn / max(tn + fn, 1)
    retained_f1 = (
        2 * (retained_precision * retained_recall) / max(retained_precision + retained_recall, 1e-9)
    )

    churn_recall = tp / max(tp + fn, 1)
    churn_precision = tp / max(tp + fp, 1)
    churn_f1 = float(f1_score(y_true, y_pred, zero_division=0))

    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None

    return {
        "threshold": round(threshold, 2),
        "balanced_accuracy": round(bal_acc, 4),
        "overall_accuracy": round(acc, 4),
        "retained_recall_tn_rate": round(retained_recall, 4),
        "retained_precision": round(retained_precision, 4),
        "retained_f1": round(retained_f1, 4),
        "churn_recall_tp_rate": round(churn_recall, 4),
        "churn_precision": round(churn_precision, 4),
        "churn_f1": round(churn_f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def run_imbalance_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tune_metric: str = "balanced_accuracy",
    model_type: str = "logistic_regression",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluate multiple class weighting and resampling strategies on test data."""
    strategies = [
        ("1. Unweighted Baseline", {"class_weight": None, "resample": None}),
        ("2. Cost-Sensitive (5:1)", {"class_weight": {0: 5.0, 1: 1.0}, "resample": None}),
        ("3. Cost-Sensitive (15:1)", {"class_weight": {0: 15.0, 1: 1.0}, "resample": None}),
        ("4. Inverse Frequency ('balanced')", {"class_weight": "balanced", "resample": None}),
        ("5. Cost-Sensitive (45:1)", {"class_weight": {0: 45.0, 1: 1.0}, "resample": None}),
        ("6. Undersampling (1:1)", {"class_weight": None, "resample": ("under", 1.0)}),
        ("7. Undersampling (3:1)", {"class_weight": None, "resample": ("under", 3.0)}),
        ("8. Oversampling (1:1)", {"class_weight": None, "resample": ("over", 1.0)}),
    ]

    results_list = []
    trained_models = {}

    y_test_arr = np.asarray(y_test).astype(int)

    for name, config in strategies:
        print(f"[{model_type.upper()}] Running Experiment: {name}...")

        X_tr = X_train.copy()
        y_tr = y_train.copy()

        # Apply resampling if configured
        if config["resample"]:
            rtype, ratio = config["resample"]
            if rtype == "under":
                X_tr, y_tr = random_undersample(X_tr, y_tr, majority_to_minority_ratio=ratio)
            elif rtype == "over":
                X_tr, y_tr = random_oversample(X_tr, y_tr, majority_to_minority_ratio=ratio)

        # Train model
        if model_type.lower() in ("lightgbm", "lgbm"):
            model = ChurnLightGBM(class_weight=config["class_weight"])
        elif model_type.lower() in ("logistic_regression", "logreg", "lr"):
            model = ChurnLogisticRegression(class_weight=config["class_weight"])
        else:
            raise ValueError(f"Unsupported model_type '{model_type}'. Use 'logistic_regression' or 'lightgbm'.")

        model.fit(X_tr, y_tr)

        # Tune threshold on validation split
        best_t, _ = model.tune_threshold(X_val, y_val, metric=tune_metric)

        # Evaluate on test split
        probs = model.predict_proba(X_test)
        preds = (probs >= best_t).astype(int)
        metrics = evaluate_strategy_predictions(y_test_arr, preds, probs, threshold=best_t)

        trained_models[name] = {"model": model, "metrics": metrics}

        cm = metrics["confusion_matrix"]
        results_list.append({
            "Strategy": name,
            "Tuned Thresh": metrics["threshold"],
            "Balanced Acc": metrics["balanced_accuracy"],
            "Retained Recall (TN%)": f"{metrics['retained_recall_tn_rate'] * 100:.1f}%",
            "Retained Prec": f"{metrics['retained_precision'] * 100:.1f}%",
            "Churn Recall (TP%)": f"{metrics['churn_recall_tp_rate'] * 100:.1f}%",
            "ROC-AUC": metrics["roc_auc"],
            "PR-AUC": metrics["pr_auc"],
            "TN": cm["tn"],
            "FP": cm["fp"],
            "FN": cm["fn"],
            "TP": cm["tp"],
        })

    results_df = pd.DataFrame(results_list)
    return results_df, trained_models


def run_lightgbm_imbalance_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tune_metric: str = "balanced_accuracy",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run imbalance experiments using LightGBM classifier."""
    return run_imbalance_experiments(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        tune_metric=tune_metric,
        model_type="lightgbm",
    )


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values]
    return "\n".join([header, sep] + rows)


def main():
    """Run imbalance analysis and benchmark suite for Logistic Regression and LightGBM."""
    print("Loading dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_model_ready_data()

    print("\nPerforming Class Imbalance Analysis...")
    analysis = analyze_class_imbalance(y_train, y_val, y_test)
    report_text = format_imbalance_report(analysis)
    print(report_text)

    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Logistic Regression Imbalance Experiments
    print("\n" + "=" * 60)
    print("RUNNING LOGISTIC REGRESSION IMBALANCE EXPERIMENTS")
    print("=" * 60)
    logreg_df, _ = run_imbalance_experiments(
        X_train, y_train, X_val, y_val, X_test, y_test, tune_metric="balanced_accuracy", model_type="logistic_regression"
    )
    print("\nLOGISTIC REGRESSION IMBALANCE EXPERIMENTS TABLE:")
    print(logreg_df.to_string(index=False))

    logreg_csv_path = reports_dir / "imbalance_experiments.csv"
    logreg_df.to_csv(logreg_csv_path, index=False)

    logreg_md_path = reports_dir / "imbalance_experiments_report.md"
    best_logreg = logreg_df.sort_values("Balanced Acc", ascending=False).iloc[0]
    with logreg_md_path.open("w", encoding="utf-8") as f:
        f.write("# Class Imbalance Experiments - Logistic Regression\n\n")
        f.write("## 1. Imbalance Analysis\n\n```text\n")
        f.write(report_text)
        f.write("\n```\n\n## 2. Benchmark Results\n\n")
        f.write(_dataframe_to_markdown(logreg_df))
        f.write("\n\n## 3. Findings & Recommendation\n\n")
        f.write(f"- **Best Strategy**: `{best_logreg['Strategy']}`\n")
        f.write(f"- **Top Balanced Accuracy**: `{best_logreg['Balanced Acc']}`\n")
        f.write(f"- **Retained Customer Detection**: `{best_logreg['Retained Recall (TN%)']}` of minority class captured.\n")

    # 2. LightGBM Imbalance Experiments
    print("\n" + "=" * 60)
    print("RUNNING LIGHTGBM IMBALANCE EXPERIMENTS")
    print("=" * 60)
    lgbm_df, _ = run_lightgbm_imbalance_experiments(
        X_train, y_train, X_val, y_val, X_test, y_test, tune_metric="balanced_accuracy"
    )
    print("\nLIGHTGBM IMBALANCE EXPERIMENTS TABLE:")
    print(lgbm_df.to_string(index=False))

    lgbm_csv_path = reports_dir / "imbalance_experiments_lightgbm.csv"
    lgbm_df.to_csv(lgbm_csv_path, index=False)

    lgbm_md_path = reports_dir / "imbalance_experiments_lightgbm_report.md"
    best_lgbm = lgbm_df.sort_values("Balanced Acc", ascending=False).iloc[0]
    with lgbm_md_path.open("w", encoding="utf-8") as f:
        f.write("# Class Imbalance Experiments - LightGBM\n\n")
        f.write("## 1. Imbalance Analysis\n\n```text\n")
        f.write(report_text)
        f.write("\n```\n\n## 2. Benchmark Results\n\n")
        f.write(_dataframe_to_markdown(lgbm_df))
        f.write("\n\n## 3. Findings & Recommendation\n\n")
        f.write(f"- **Best Strategy**: `{best_lgbm['Strategy']}`\n")
        f.write(f"- **Top Balanced Accuracy**: `{best_lgbm['Balanced Acc']}`\n")
        f.write(f"- **Retained Customer Detection**: `{best_lgbm['Retained Recall (TN%)']}` of minority class captured.\n")

    print(f"\nResults successfully exported to:")
    print(f"  - Logistic Regression CSV: {logreg_csv_path}")
    print(f"  - Logistic Regression MD:  {logreg_md_path}")
    print(f"  - LightGBM CSV:            {lgbm_csv_path}")
    print(f"  - LightGBM MD:             {lgbm_md_path}")


if __name__ == "__main__":
    main()
