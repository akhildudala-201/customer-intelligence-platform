
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
APP_ROOT = Path(__file__).resolve().parent

for candidate in (str(PROJECT_ROOT), str(APP_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from app.ml.metrics import calculate_metrics, find_optimal_threshold, format_metrics_summary
except ModuleNotFoundError:
    from ml.metrics import calculate_metrics, find_optimal_threshold, format_metrics_summary

try:
    from app.ml.experiment_logger import log_experiment
except ModuleNotFoundError:
    from ml.experiment_logger import log_experiment

try:
    from app.ml.logistic_regression import ChurnLogisticRegression, load_model_ready_data
except ModuleNotFoundError:
    from ml.logistic_regression import ChurnLogisticRegression, load_model_ready_data


METRICS_TO_COMPARE = [
    "roc_auc", "pr_auc", "balanced_accuracy",
    "precision", "recall", "f1", "f2", "log_loss",
]

# The business target both models must agree on: churn_label == CHURN_TARGET_VALUE
# means "churned." Logistic Regression already predicts P(churn_label == 1) directly.
CHURN_TARGET_VALUE = 1


# ---------------------------------------------------------------------------
# LightGBM artifact adapter
# ---------------------------------------------------------------------------

class LoadedLightGBMModel:
    """Wraps the artifact dict produced by train_lightgbm_model.py's save_model():
    {"model": <raw estimator>, "feature_cols": [...], "positive_class": ..., ...}

    Always returns predict_proba() oriented as P(churn_label == CHURN_TARGET_VALUE),
    regardless of which original label the underlying model was actually trained
    to treat as its positive class (1)."""

    def __init__(self, artifact: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        self.raw_model = artifact["model"]  # lgb.LGBMClassifier, or imblearn Pipeline if SMOTE was used
        self.feature_names_: List[str] = list(
            artifact.get("feature_cols")
            or (metadata or {}).get("feature_cols")
            or []
        )
        self.source_positive_class = artifact.get(
            "positive_class", (metadata or {}).get("positive_class", CHURN_TARGET_VALUE)
        )
        # If the model was trained to predict original_label == source_positive_class,
        # and that's not the same as our churn target, its raw proba needs inverting.
        self.invert_proba = (self.source_positive_class != CHURN_TARGET_VALUE)
        if self.invert_proba:
            print(
                f"  NOTE: LightGBM artifact was trained with positive_class="
                f"{self.source_positive_class} (auto-selected minority label), not "
                f"{CHURN_TARGET_VALUE}. Inverting probabilities so predict_proba() "
                f"returns P(churn_label == {CHURN_TARGET_VALUE}), matching Logistic Regression."
            )

        self.threshold: float = 0.5
        self.metrics_: Dict[str, Any] = {}
        self.is_fitted_ = True

        op = artifact.get("operating_point") or (metadata or {}).get("operating_point")
        if op and op.get("mode") == "score":
            t = float(op["value"])
            # The stored score threshold was chosen against the model's ORIGINAL
            # (possibly inverted) probability. If we invert probabilities, the
            # equivalent threshold on the inverted scale is (1 - t).
            self.threshold = (1.0 - t) if self.invert_proba else t

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_:
            missing = [c for c in self.feature_names_ if c not in X.columns]
            if missing:
                raise ValueError(f"LightGBM artifact expects columns not present in X: {missing}")
            return X[self.feature_names_]
        return X

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = self._align_columns(X) if isinstance(X, pd.DataFrame) else X
        raw_proba = self.raw_model.predict_proba(X_aligned)[:, 1]
        return (1.0 - raw_proba) if self.invert_proba else raw_proba

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        t = self.threshold if threshold is None else threshold
        return (self.predict_proba(X) >= t).astype(int)

    def tune_threshold(self, X_val, y_val, metric: str = "f1", thresholds=None) -> Tuple[float, float]:
        probs = self.predict_proba(X_val)
        best_t, best_score = find_optimal_threshold(y_val, probs, metric=metric, thresholds=thresholds)
        self.threshold = best_t
        return best_t, best_score

    def evaluate(self, X, y, threshold: Optional[float] = None) -> Dict[str, Any]:
        t = self.threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        preds = (probs >= t).astype(int)
        metrics = calculate_metrics(y_true=y, y_pred=preds, y_prob=probs, threshold=t)
        self.metrics_ = metrics
        return metrics


def load_logreg_model(path: Path) -> ChurnLogisticRegression:
    """Load a trained ChurnLogisticRegression artifact. Does NOT train."""
    print(f"Loading Logistic Regression artifact: {path}")
    return ChurnLogisticRegression.load(path)


def load_lightgbm_model(path: Path, metadata_path: Optional[Path] = None) -> LoadedLightGBMModel:
    """Load a trained LightGBM artifact. Does NOT train."""
    print(f"Loading LightGBM artifact: {path}")
    artifact = joblib.load(path)

    metadata = None
    if metadata_path and Path(metadata_path).exists():
        print(f"Loading LightGBM metadata: {metadata_path}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    return LoadedLightGBMModel(artifact, metadata)


def load_val_test_data():
    """Loads the same val/test splits/features both models were originally trained on."""
    _, _, X_val, y_val, X_test, y_test = load_model_ready_data()
    return X_val, y_val, X_test, y_test


def find_latest_lightgbm_artifact(models_dir: Path) -> Path:
    """Find the newest lgb_churn_model_<timestamp>.joblib training artifact.

    Excludes any '*_calibrated.joblib' files (those are calibration.py's
    output, a different schema) so this never picks up the wrong artifact.
    Mirrors threshold_analysis.py's helper of the same name.
    """
    candidates = sorted(
        p for p in models_dir.glob("lgb_churn_model_*.joblib")
        if "_calibrated" not in p.stem
    )
    if not candidates:
        raise FileNotFoundError(
            f"No lgb_churn_model_*.joblib found in {models_dir}. "
            f"Run train_lightgbm_model.py first to produce one."
        )
    return candidates[-1]


def guess_metadata_path(lgbm_path: Path, models_dir: Path) -> Optional[Path]:
    """Derive the matching metadata_<timestamp>.json path from a LightGBM artifact path."""
    ts = lgbm_path.stem.replace("lgb_churn_model_", "")
    guess = models_dir / f"metadata_{ts}.json"
    return guess if guess.exists() else None


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def tune_and_evaluate(model, X_val, y_val, X_test, y_test, tune_metric: str) -> Dict[str, Any]:
    """Tune threshold on validation data, then evaluate on the held-out test set."""
    best_t, best_val_score = model.tune_threshold(X_val, y_val, metric=tune_metric)
    test_metrics = model.evaluate(X_test, y_test, threshold=best_t)
    val_metrics = model.evaluate(X_val, y_val, threshold=best_t)
    return {
        "threshold": best_t,
        "val_score_at_tuning": best_val_score,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def build_comparison_table(logreg_result: Dict[str, Any], lgbm_result: Dict[str, Any]) -> pd.DataFrame:
    """Winner per metric is computed from the actual numbers -- never assumed."""
    rows = []
    for metric in METRICS_TO_COMPARE:
        lr_val = logreg_result["test_metrics"].get(metric)
        lgb_val = lgbm_result["test_metrics"].get(metric)
        if lr_val is None or lgb_val is None:
            winner = "n/a"
        elif metric == "log_loss":  # lower is better
            winner = "LightGBM" if lgb_val < lr_val else ("Logistic Regression" if lr_val < lgb_val else "tie")
        else:  # higher is better
            winner = "LightGBM" if lgb_val > lr_val else ("Logistic Regression" if lr_val > lgb_val else "tie")
        rows.append({
            "metric": metric,
            "logistic_regression": lr_val,
            "lightgbm": lgb_val,
            "better": winner,
        })
    return pd.DataFrame(rows)


def summarize_winner(comparison_df: pd.DataFrame) -> str:
    counts = comparison_df["better"].value_counts()
    lines = ["Metric wins:"]
    for model_name, n in counts.items():
        lines.append(f"  {model_name}: {n}/{len(comparison_df)} metrics")
    return "\n".join(lines)


def export_probabilities(
    model, X: pd.DataFrame, y: pd.Series, split_name: str, model_name: str, out_dir: Path
) -> pd.DataFrame:
    """Save per-row predicted probabilities (of churn=1) alongside true labels."""
    probs = model.predict_proba(X)
    df = pd.DataFrame({
        "row_index": X.index,
        "y_true": np.asarray(y),
        "y_prob": probs,
        "model": model_name,
        "split": split_name,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"probabilities_{model_name}_{split_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {model_name} {split_name} probabilities to: {out_path}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare pre-trained Logistic Regression vs LightGBM churn models.")
    parser.add_argument(
        "--logreg-path", default=str(PROJECT_ROOT / "outputs" / "models" / "churn_logistic_regression.joblib"),
        help="Path to the trained ChurnLogisticRegression .joblib artifact.",
    )
    parser.add_argument(
        "--lgbm-path",
        default=None,
        help="Path to the trained LightGBM .joblib artifact. "
             "If omitted, auto-detects the newest lgb_churn_model_*.joblib in outputs/models/.",
    )
    parser.add_argument(
        "--lgbm-metadata",
        default=None,
        help="Optional path to the matching metadata_<timestamp>.json. "
             "If omitted, auto-derived from --lgbm-path (or its auto-detected value).",
    )
    parser.add_argument(
        "--tune-metric", default="balanced_accuracy",
        choices=["f1", "f2", "balanced_accuracy", "precision", "recall", "accuracy"],
        help="Validation metric used to select each model's decision threshold.",
    )
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "outputs" / "reports"),
        help="Directory to write the comparison table and probability CSVs.",
    )
    parser.add_argument("--log-run", action="store_true", help="Log both runs to the centralized experiment log.")
    args = parser.parse_args()

    models_dir = PROJECT_ROOT / "outputs" / "models"

    # Auto-detect the newest LightGBM training artifact if not explicitly given.
    if args.lgbm_path is None:
        latest = find_latest_lightgbm_artifact(models_dir)
        args.lgbm_path = str(latest)
        print(f"No --lgbm-path given; using latest: {args.lgbm_path}")

    # Auto-derive the matching metadata file if not explicitly given.
    if args.lgbm_metadata is None:
        guessed = guess_metadata_path(Path(args.lgbm_path), models_dir)
        args.lgbm_metadata = str(guessed) if guessed else None

    print("Loading val/test data (same splits/features both models were trained on)...")
    X_val, y_val, X_test, y_test = load_val_test_data()
    print(f"  Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"  Test churn rate (label={CHURN_TARGET_VALUE}): {y_test.mean():.2%}")

    logreg = load_logreg_model(Path(args.logreg_path))
    lgbm = load_lightgbm_model(Path(args.lgbm_path), Path(args.lgbm_metadata) if args.lgbm_metadata else None)

    # Both models now predict the SAME target (churn_label == 1) -- no label
    # remapping needed. y_val / y_test are used as-is for both.
    print(f"\nTuning thresholds on validation set (metric = {args.tune_metric})...")
    logreg_result = tune_and_evaluate(logreg, X_val, y_val, X_test, y_test, args.tune_metric)
    lgbm_result = tune_and_evaluate(lgbm, X_val, y_val, X_test, y_test, args.tune_metric)

    print(f"  Logistic Regression tuned threshold: {logreg_result['threshold']}")
    print(f"  LightGBM tuned threshold:             {lgbm_result['threshold']}")

    print("\n" + format_metrics_summary(logreg_result["test_metrics"], title="LOGISTIC REGRESSION — TEST SET"))
    print("\n" + format_metrics_summary(lgbm_result["test_metrics"], title="LIGHTGBM — TEST SET"))

    comparison_df = build_comparison_table(logreg_result, lgbm_result)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (test set)")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("\n" + summarize_winner(comparison_df))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = out_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\nSaved comparison table to: {comparison_csv}")

    print("\nExporting per-row predicted probabilities...")
    export_probabilities(logreg, X_val, y_val, "val", "LogisticRegression", out_dir)
    export_probabilities(logreg, X_test, y_test, "test", "LogisticRegression", out_dir)
    export_probabilities(lgbm, X_val, y_val, "val", "LightGBM", out_dir)
    export_probabilities(lgbm, X_test, y_test, "test", "LightGBM", out_dir)

    if args.log_run:
        try:
            log_experiment(
                model_name="LogisticRegression",
                strategy=f"loaded artifact (tuned on {args.tune_metric})",
                metrics=logreg_result["test_metrics"],
                features_count=len(logreg.feature_names_),
                threshold=logreg_result["threshold"],
                artifacts_path=args.logreg_path,
            )
            log_experiment(
                model_name="LightGBM",
                strategy=f"loaded artifact (tuned on {args.tune_metric}, proba-oriented to churn=1)",
                metrics=lgbm_result["test_metrics"],
                features_count=len(lgbm.feature_names_),
                threshold=lgbm_result["threshold"],
                artifacts_path=args.lgbm_path,
            )
            print("\nLogged both runs to the centralized experiment log.")
        except Exception as exc:
            print(f"Could not log experiment runs: {exc}")

    return comparison_df


if __name__ == "__main__":
    main()