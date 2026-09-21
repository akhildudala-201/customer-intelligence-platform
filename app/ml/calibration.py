"""Probability calibration, calibration evaluation, and final test evaluation.

This module takes an already-trained LightGBM classifier (as produced by
Person 2's pipeline) and:
    1. Diagnoses how well-calibrated its raw probabilities are.
    2. Fits two calibration methods (Platt / sigmoid scaling and Isotonic
       regression) on the VALIDATION split (never train, to avoid leakage
       into an already-fitted model).
    3. Picks the best calibration method by Brier score on validation.
    4. Re-tunes the decision threshold on the calibrated validation
       probabilities (calibration shifts the probability scale, so the
       pre-calibration threshold is no longer valid). The METRIC used for
       this re-tuning is read from Person 4's threshold_analysis.py output
       (outputs/reports/best_threshold_comparison_val.csv or
       best_thresholds_LightGBM_val.csv) -- this module does not decide the
       tuning objective independently, it defers to Person 4's analysis,
       with a safe hardcoded fallback if that analysis hasn't been run yet.
    5. Produces a final, honest TEST-set evaluation with the calibrated
       model + new threshold.
    6. Saves a single deployable artifact (base model + calibrator +
       threshold) for Person 6's predict() / churn_predictions integration.
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[1]

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
    from app.ml.logistic_regression import load_model_ready_data
except ModuleNotFoundError:
    from ml.logistic_regression import load_model_ready_data

# Metrics find_optimal_threshold() actually supports (see metrics.py's
# metric_funcs dict). Used to validate whatever metric name we read back
# from Person 4's threshold analysis output before trusting it.
SUPPORTED_TUNE_METRICS = {"f1", "f2", "balanced_accuracy", "precision", "recall", "accuracy"}

CONFIG = {
    "N_BINS": 10,  # bins for reliability diagrams / ECE
    "THRESHOLD_TUNE_METRIC_DEFAULT": "balanced_accuracy",  # fallback only
    "OUTPUT_MODEL_DIR": str(PROJECT_ROOT / "outputs" / "models") + "/",
    "OUTPUT_REPORT_DIR": str(PROJECT_ROOT / "outputs" / "reports") + "/",
    "TIMESTAMP": datetime.now().strftime("%Y%m%d_%H%M%S"),
}

Path(CONFIG["OUTPUT_MODEL_DIR"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["OUTPUT_REPORT_DIR"]).mkdir(parents=True, exist_ok=True)


def log_message(message: str, level: str = "INFO") -> None:
    """Format and print application log message with timestamp."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{level}] {message}")


def banner(text: str) -> None:
    """Print standard section banner."""
    log_message("=" * 60)
    log_message(text)
    log_message("=" * 60)


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Bins predictions into equal-width probability bins, then measures the
    weighted (ECE) and worst-case (MCE) gap between predicted confidence
    and observed accuracy/frequency of the positive class in each bin.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(y_prob)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        bin_count = int(mask.sum())
        if bin_count == 0:
            continue

        bin_confidence = float(y_prob[mask].mean())
        bin_accuracy = float(y_true[mask].mean())
        gap = abs(bin_confidence - bin_accuracy)

        ece += (bin_count / n) * gap
        mce = max(mce, gap)

    return round(ece, 4), round(mce, 4)


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Compute Brier score, log loss, ECE, and MCE for a set of probabilities."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    eps = 1e-15
    clipped = np.clip(y_prob, eps, 1 - eps)

    ece, mce = expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    return {
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "log_loss": round(float(log_loss(y_true, clipped)), 4),
        "ece": ece,
        "mce": mce,
    }


def plot_reliability_diagram(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str,
    save_path: Union[str, Path],
    n_bins: int = 10,
) -> None:
    """Plot reliability diagram(s) comparing predicted probability vs observed frequency.

    Args:
        curves: mapping of {label: (y_true, y_prob)} to overlay on one plot.
        title: plot title.
        save_path: output PNG path.
        n_bins: number of bins for the calibration curve.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", linewidth=1)
    for label, (y_true, y_prob) in curves.items():
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        ax1.plot(mean_pred, frac_pos, marker="o", linewidth=2, label=label)

    ax1.set_ylabel("Observed frequency (actual churn rate)")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    for label, (_, y_prob) in curves.items():
        ax2.hist(y_prob, bins=20, range=(0, 1), alpha=0.5, label=label)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    log_message(f"Saved reliability diagram: {save_path}")


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------

class PlattScaling:
    """Sigmoid (Platt) probability recalibration via 1-D logistic regression."""

    def __init__(self):
        self.lr = LogisticRegression()
        self.is_fitted_ = False

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "PlattScaling":
        probs = np.asarray(probs).astype(float).reshape(-1, 1)
        self.lr.fit(probs, np.asarray(y).astype(int))
        self.is_fitted_ = True
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("PlattScaling is not fitted. Call fit() first.")
        probs = np.asarray(probs).astype(float).reshape(-1, 1)
        return self.lr.predict_proba(probs)[:, 1]


class IsotonicCalibration:
    """Non-parametric monotonic probability recalibration via isotonic regression."""

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.is_fitted_ = False

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "IsotonicCalibration":
        self.iso.fit(np.asarray(probs).astype(float), np.asarray(y).astype(int))
        self.is_fitted_ = True
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("IsotonicCalibration is not fitted. Call fit() first.")
        return self.iso.predict(np.asarray(probs).astype(float))


class CalibratedChurnModel:
    """Wraps a fitted base churn model with a probability calibrator and threshold.

    This is the artifact Person 6 should load for predict() / churn_predictions:
    it exposes the same predict_proba()/predict() interface as the base models,
    but returns calibrated probabilities and applies the recalibrated threshold.
    """

    # Matches build_churn_label.py's / _LGBAdapter's convention: label 1 =
    # churned. Only used to annotate the saved artifact's metadata (see
    # save()) -- orientation itself is enforced upstream by _LGBAdapter,
    # not derived from this constant.
    CHURN_TARGET_LABEL = 1

    def __init__(
        self,
        base_model: Any,
        calibrator: Optional[Union[PlattScaling, IsotonicCalibration]] = None,
        method: str = "none",
        threshold: float = 0.5,
        model_name: str = "unknown",
        feature_cols: Optional[list] = None,
    ):
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method  # 'none' | 'sigmoid' | 'isotonic'
        self.threshold = threshold
        self.model_name = model_name
        # Carried through from the base model's own artifact (via
        # _LGBAdapter.feature_cols) so the final saved artifact still
        # records its feature_cols -- without this, feature_contract.py's
        # check_contract_matches_model() has nothing to compare against
        # once this artifact is loaded, and silently skips the check.
        self.feature_cols = list(feature_cols) if feature_cols else list(
            getattr(base_model, "feature_cols", []) or []
        )

    def raw_predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Uncalibrated probabilities straight from the base model."""
        return self.base_model.predict_proba(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Calibrated positive-class probabilities."""
        raw = self.raw_predict_proba(X)
        if self.calibrator is None:
            return raw
        return self.calibrator.predict_proba(raw)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        t = self.threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        return (probs >= t).astype(int)

    def save(self, filepath: Union[str, Path]) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "base_model": self.base_model,
                "calibrator": self.calibrator,
                "method": self.method,
                "threshold": self.threshold,
                "model_name": self.model_name,
                "feature_cols": self.feature_cols,
                # Informational only -- not read back by load() or by
                # model_adapter.py, which derives orientation itself from
                # base_model (_LGBAdapter)'s own positive_class. These
                # exist so anyone inspecting the artifact directly (a
                # notebook, a debugging session, a future Person 6 change)
                # doesn't have to reverse-engineer what predict_proba()
                # actually means from the code.
                "target_value": self.CHURN_TARGET_LABEL,
                "probability_definition": (
                    f"P(churn_label == {self.CHURN_TARGET_LABEL})"
                ),
                "source_positive_class": getattr(self.base_model, "positive_class", None),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CalibratedChurnModel":
        bundle = joblib.load(Path(filepath))
        return cls(
            base_model=bundle["base_model"],
            calibrator=bundle.get("calibrator"),
            method=bundle.get("method", "none"),
            threshold=bundle.get("threshold", 0.5),
            model_name=bundle.get("model_name", "unknown"),
            feature_cols=bundle.get("feature_cols"),
        )


# ---------------------------------------------------------------------------
# Reading Person 4's threshold analysis output (threshold_analysis.py)
# ---------------------------------------------------------------------------

def load_recommended_tune_metric(
    reports_dir: Path,
    default_metric: str = "balanced_accuracy",
) -> str:
    """Determine which metric to re-tune the post-calibration threshold against.

    This defers to Person 4's threshold_analysis.py output rather than
    hardcoding a choice:
      1. Prefer outputs/reports/best_threshold_comparison_val.csv -- pick the
         metric where LightGBM beats Logistic Regression by the widest margin
         (column 'better' == 'LightGBM', ranked by 'score_gap'). This reflects
         Person 4's head-to-head model comparison, not just LightGBM in isolation.
      2. Fall back to outputs/reports/best_thresholds_LightGBM_val.csv -- pick
         the metric with LightGBM's highest achievable validation score.
      3. If neither file exists yet (threshold_analysis.py hasn't been run),
         fall back to `default_metric` with a warning.

    Any metric name recovered from disk is validated against
    SUPPORTED_TUNE_METRICS before use, in case that file ever contains a
    metric find_optimal_threshold() doesn't recognize.
    """
    comparison_path = reports_dir / "best_threshold_comparison_val.csv"
    lgbm_only_path = reports_dir / "best_thresholds_LightGBM_val.csv"

    if comparison_path.exists():
        try:
            df = pd.read_csv(comparison_path)
            lgbm_wins = df[df["better"] == "LightGBM"]
            if not lgbm_wins.empty:
                best_row = lgbm_wins.loc[lgbm_wins["score_gap"].idxmax()]
                metric = str(best_row["metric"])
                if metric in SUPPORTED_TUNE_METRICS:
                    log_message(
                        f"Tune metric sourced from Person 4's threshold analysis "
                        f"('{comparison_path.name}'): '{metric}' "
                        f"(LightGBM leads by {best_row['score_gap']:.4f} on validation)."
                    )
                    return metric
                log_message(
                    f"'{comparison_path.name}' recommended metric '{metric}', which "
                    f"find_optimal_threshold() doesn't support. Falling back.",
                    "WARNING",
                )
            else:
                log_message(
                    f"'{comparison_path.name}' found, but LightGBM doesn't lead on any "
                    f"metric there. Falling back to LightGBM's own best-scoring metric.",
                    "WARNING",
                )
        except Exception as exc:
            log_message(f"Could not read '{comparison_path.name}': {exc}", "WARNING")

    if lgbm_only_path.exists():
        try:
            df = pd.read_csv(lgbm_only_path)
            best_row = df.loc[df["best_score"].idxmax()]
            metric = str(best_row["metric"])
            if metric in SUPPORTED_TUNE_METRICS:
                log_message(
                    f"Tune metric sourced from Person 4's threshold analysis "
                    f"('{lgbm_only_path.name}'): '{metric}' "
                    f"(best_score={best_row['best_score']:.4f} on validation)."
                )
                return metric
            log_message(
                f"'{lgbm_only_path.name}' recommended metric '{metric}', which "
                f"find_optimal_threshold() doesn't support. Falling back.",
                "WARNING",
            )
        except Exception as exc:
            log_message(f"Could not read '{lgbm_only_path.name}': {exc}", "WARNING")

    log_message(
        f"No threshold analysis output found in '{reports_dir}' "
        f"(expected best_threshold_comparison_val.csv or best_thresholds_LightGBM_val.csv "
        f"from threshold_analysis.py). Run Person 4's script first for a data-driven "
        f"choice. Falling back to default tune metric: '{default_metric}'.",
        "WARNING",
    )
    return default_metric


# ---------------------------------------------------------------------------
# Calibration workflow
# ---------------------------------------------------------------------------

def compare_calibration_methods(
    base_model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_bins: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Optional[Union[PlattScaling, IsotonicCalibration]]]]:
    """Fit sigmoid and isotonic calibrators on validation data, score all three variants.

    Returns a comparison table (uncalibrated / sigmoid / isotonic) scored by
    Brier score, log loss, ECE and MCE — all on the SAME validation split the
    calibrators were fit on (this is a diagnostic comparison; the honest,
    unbiased final numbers come from evaluate_on_test() on the held-out test set).
    """
    y_val_arr = np.asarray(y_val).astype(int)
    raw_val_probs = base_model.predict_proba(X_val)

    platt = PlattScaling().fit(raw_val_probs, y_val_arr)
    isotonic = IsotonicCalibration().fit(raw_val_probs, y_val_arr)

    variants = {
        "none": raw_val_probs,
        "sigmoid": platt.predict_proba(raw_val_probs),
        "isotonic": isotonic.predict_proba(raw_val_probs),
    }
    calibrators = {"none": None, "sigmoid": platt, "isotonic": isotonic}

    rows = []
    for name, probs in variants.items():
        m = calibration_metrics(y_val_arr, probs, n_bins=n_bins)
        rows.append({"method": name, **m})

    comparison_df = pd.DataFrame(rows).sort_values("brier_score").reset_index(drop=True)
    return comparison_df, calibrators


def calibrate_model(
    base_model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str = "model",
    n_bins: int = 10,
    tune_metric: str = "balanced_accuracy",
) -> Tuple[CalibratedChurnModel, pd.DataFrame]:
    """Run the full calibration workflow for one base model.

    1. Compare uncalibrated / sigmoid / isotonic on validation (by Brier score).
    2. Select the best method (falls back to 'none' if calibration doesn't help).
    3. Re-tune the decision threshold on the CALIBRATED validation probabilities,
       against `tune_metric` (sourced from Person 4's threshold analysis by the
       caller -- see load_recommended_tune_metric()).
    4. Return a ready-to-use CalibratedChurnModel.

    METHODOLOGY NOTE: both the calibrators (step 1) and the method-selection
    decision (step 2) use the SAME validation split -- Platt/Isotonic are fit
    on (X_val, y_val), then compared against each other on (X_val, y_val)
    again. This is standard for calibration (calibrators need to be fit on
    data the base model wasn't trained on, which val already satisfies), but
    choosing the "best" method by its own in-sample validation score is a
    mild form of model-selection overfitting -- it doesn't leak into the
    base LightGBM model, but it can optimistically favor whichever method
    happens to fit validation-set noise slightly better. evaluate_on_test()
    reports final metrics on the untouched TEST split specifically so this
    optimism doesn't reach the final reported numbers. A stricter approach
    (cross-validated calibrator selection) would remove even this mild bias,
    at the cost of real added complexity -- left as a known limitation
    rather than implemented, given the scope of this project.
    """
    banner(f"CALIBRATING: {model_name}")

    comparison_df, calibrators = compare_calibration_methods(base_model, X_val, y_val, n_bins=n_bins)
    log_message("Validation calibration comparison (lower brier/log_loss/ece/mce is better):")
    log_message("\n" + comparison_df.to_string(index=False))

    best_method = comparison_df.iloc[0]["method"]
    uncalibrated_brier = comparison_df.loc[comparison_df["method"] == "none", "brier_score"].values[0]
    best_brier = comparison_df.iloc[0]["brier_score"]

    if best_method != "none" and (uncalibrated_brier - best_brier) < 1e-4:
        log_message(
            f"Calibration improvement is negligible ({uncalibrated_brier} -> {best_brier}); "
            f"keeping uncalibrated probabilities.",
            "WARNING",
        )
        best_method = "none"

    log_message(f"Selected calibration method: '{best_method}'")
    chosen_calibrator = calibrators[best_method]

    y_val_arr = np.asarray(y_val).astype(int)
    raw_val_probs = base_model.predict_proba(X_val)
    calibrated_val_probs = (
        raw_val_probs if chosen_calibrator is None else chosen_calibrator.predict_proba(raw_val_probs)
    )

    log_message(f"Re-tuning threshold against metric: '{tune_metric}' (from Person 4's analysis).")
    best_t, best_score = find_optimal_threshold(y_val_arr, calibrated_val_probs, metric=tune_metric)
    log_message(f"Re-tuned threshold on calibrated validation probs: {best_t} ({tune_metric}={best_score})")

    calibrated_model = CalibratedChurnModel(
        base_model=base_model,
        calibrator=chosen_calibrator,
        method=best_method,
        threshold=best_t,
        model_name=model_name,
    )
    return calibrated_model, comparison_df


def evaluate_on_test(
    calibrated_model: CalibratedChurnModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Final, unbiased evaluation on the held-out test set: classification + calibration metrics."""
    banner(f"FINAL TEST EVALUATION: {calibrated_model.model_name} ({calibrated_model.method})")

    y_test_arr = np.asarray(y_test).astype(int)
    raw_test_probs = calibrated_model.raw_predict_proba(X_test)
    calibrated_test_probs = calibrated_model.predict_proba(X_test)
    preds = calibrated_model.predict(X_test)

    classification_metrics = calculate_metrics(
        y_true=y_test_arr,
        y_pred=preds,
        y_prob=calibrated_test_probs,
        threshold=calibrated_model.threshold,
    )
    print("\n" + format_metrics_summary(
        classification_metrics, title=f"TEST SET - {calibrated_model.model_name} (calibrated)"
    ))

    cal_metrics_before = calibration_metrics(y_test_arr, raw_test_probs, n_bins=n_bins)
    cal_metrics_after = calibration_metrics(y_test_arr, calibrated_test_probs, n_bins=n_bins)

    log_message("Calibration quality on TEST set (uncalibrated vs calibrated):")
    log_message(f"   Brier Score : {cal_metrics_before['brier_score']} -> {cal_metrics_after['brier_score']}")
    log_message(f"   Log Loss    : {cal_metrics_before['log_loss']} -> {cal_metrics_after['log_loss']}")
    log_message(f"   ECE         : {cal_metrics_before['ece']} -> {cal_metrics_after['ece']}")
    log_message(f"   MCE         : {cal_metrics_before['mce']} -> {cal_metrics_after['mce']}")

    plot_path = (
        f"{CONFIG['OUTPUT_REPORT_DIR']}reliability_{calibrated_model.model_name}_"
        f"{CONFIG['TIMESTAMP']}.png"
    )
    plot_reliability_diagram(
        curves={
            "Uncalibrated": (y_test_arr, raw_test_probs),
            f"Calibrated ({calibrated_model.method})": (y_test_arr, calibrated_test_probs),
        },
        title=f"Reliability Diagram - {calibrated_model.model_name} (Test Set)",
        save_path=plot_path,
        n_bins=n_bins,
    )

    return {
        "classification_metrics": classification_metrics,
        "calibration_before": cal_metrics_before,
        "calibration_after": cal_metrics_after,
        "threshold": calibrated_model.threshold,
        "method": calibrated_model.method,
        "reliability_plot": plot_path,
    }


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

class _LGBAdapter:
    """Wraps a raw sklearn/lightgbm estimator so it exposes the same
    predict_proba(X) -> 1D array interface as ChurnLightGBM, already
    oriented to P(churn_label == CHURN_TARGET_VALUE) regardless of which
    original label train_lightgbm_model.py's auto-selected POSITIVE_CLASS
    resolved to for this training run.

    Without this, if POSITIVE_CLASS auto-detection picked the minority
    label (very likely 0/"retained" on this dataset, since churn_label=1
    is the majority class — see model_comparison.py's module docstring
    for the same issue), the raw model's predict_proba(X)[:, 1] would be
    P(retained), and everything downstream (calibration, threshold
    tuning, the final saved artifact) would silently calibrate/tune/serve
    the WRONG target. Correcting it once, here, at calibration time means
    the final CalibratedChurnModel artifact is guaranteed to already be
    oriented correctly — nothing downstream (model_adapter.py) needs to
    re-check or re-invert it.
    """

    CHURN_TARGET_VALUE = 1

    def __init__(self, sk_model, feature_cols, positive_class=None):
        self.sk_model = sk_model
        self.feature_cols = feature_cols
        self.positive_class = (
            positive_class if positive_class is not None else self.CHURN_TARGET_VALUE
        )
        self.invert_proba = self.positive_class != self.CHURN_TARGET_VALUE
        if self.invert_proba:
            print(
                f"  NOTE: LightGBM artifact was trained with positive_class="
                f"{self.positive_class}, not {self.CHURN_TARGET_VALUE}. Inverting "
                f"probabilities before calibration so the final artifact returns "
                f"P(churn_label == {self.CHURN_TARGET_VALUE})."
            )

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_cols].values
        raw = self.sk_model.predict_proba(X)[:, 1]
        return (1.0 - raw) if self.invert_proba else raw


def load_latest_lightgbm_model() -> Optional[Any]:
    """Load the most recently saved LightGBM TRAINING artifact from outputs/models/, if any.

    Excludes '*_calibrated.joblib' files (this module's own output) so that
    re-running calibration never tries to load its own previous output back
    in as if it were a fresh base model. Corrupt or wrong-schema files are
    skipped with a warning rather than crashing the whole pipeline -- but a
    missing or invalid `positive_class` on an otherwise well-formed
    artifact raises immediately rather than being skipped or defaulted,
    since there's no safe way to guess probability orientation (see below).
    """
    model_dir = Path(CONFIG["OUTPUT_MODEL_DIR"])
    candidates = sorted(
        p for p in model_dir.glob("lgb_churn_model_*.joblib")
        if "_calibrated" not in p.stem
    )
    if not candidates:
        log_message(
            "No saved LightGBM training artifact found in outputs/models/ "
            "(expected lgb_churn_model_<timestamp>.joblib from train_lightgbm_model.py). "
            "Run Person 2's LightGBM pipeline first if you want it calibrated too.",
            "WARNING",
        )
        return None

    for candidate in reversed(candidates):  # newest timestamp first
        try:
            bundle = joblib.load(candidate)
        except Exception as exc:
            log_message(f"Skipping unreadable/corrupt artifact '{candidate.name}': {exc}", "WARNING")
            continue

        if not isinstance(bundle, dict) or "model" not in bundle or "feature_cols" not in bundle:
            log_message(f"Skipping '{candidate.name}': unexpected artifact schema.", "WARNING")
            continue

        # positive_class should always be present -- train_lightgbm_model.py's
        # save_model() always writes it (auto-detected or explicit, never
        # None). A missing OR invalid value means this artifact's true
        # probability orientation can't be determined: there is no safe
        # default to fall back to, because guessing wrong here means every
        # prediction this artifact ever produces is silently inverted
        # (churn scored as retention and vice versa) with no visible
        # symptom until someone notices the business numbers look wrong.
        # Fail loudly instead of guessing.
        positive_class = bundle.get("positive_class")
        if positive_class is None:
            raise ValueError(
                f"'{candidate.name}' does not contain 'positive_class'. "
                f"Cannot safely determine probability orientation. "
                f"Retrain with train_lightgbm_model.py so the artifact "
                f"records positive_class explicitly."
            )
        if positive_class not in (0, 1):
            raise ValueError(
                f"'{candidate.name}' has positive_class={positive_class!r}, expected "
                f"0 or 1. Refusing to guess an orientation for an invalid value -- "
                f"fix or regenerate this artifact."
            )

        log_message(f"Loading latest LightGBM artifact: {candidate}")
        return _LGBAdapter(bundle["model"], bundle["feature_cols"], positive_class)

    log_message("No valid LightGBM training artifact could be loaded.", "WARNING")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Calibrate the LightGBM model only, evaluate, and save the final artifact.

    LightGBM was selected as the production model (it outperforms the
    Logistic Regression baseline on every metric — F1, ROC-AUC, PR-AUC —
    after calibration), so this pipeline calibrates LightGBM exclusively.

    The threshold re-tuning objective is sourced from Person 4's
    threshold_analysis.py output rather than hardcoded here -- see
    load_recommended_tune_metric().
    """
    banner("CALIBRATION PIPELINE (LightGBM only)")

    log_message("Loading data from database...")
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_model_ready_data()
    except Exception as e:
        log_message(f"Could not load data from database: {e}", "ERROR")
        sys.exit(1)

    lgbm_model = load_latest_lightgbm_model()
    if lgbm_model is None:
        log_message(
            "No LightGBM training artifact found in outputs/models/. "
            "Run Person 2's train_lightgbm_model.py first, then re-run calibration.",
            "ERROR",
        )
        sys.exit(1)

    tune_metric = load_recommended_tune_metric(
        reports_dir=Path(CONFIG["OUTPUT_REPORT_DIR"]),
        default_metric=CONFIG["THRESHOLD_TUNE_METRIC_DEFAULT"],
    )

    log_message(
        f"Validation churn prevalence: {y_val.mean():.2%} | "
        f"Test churn prevalence: {y_test.mean():.2%}"
    )

    lgbm_calibrated, lgbm_comparison = calibrate_model(
        lgbm_model, X_val, y_val, model_name="LightGBM",
        n_bins=CONFIG["N_BINS"], tune_metric=tune_metric,
    )

    comparison_csv_path = Path(CONFIG["OUTPUT_REPORT_DIR"]) / f"calibration_comparison_{CONFIG['TIMESTAMP']}.csv"
    lgbm_comparison.to_csv(comparison_csv_path, index=False)
    log_message(f"Saved calibration method comparison to: {comparison_csv_path}")

    lgbm_test_results = evaluate_on_test(lgbm_calibrated, X_test, y_test, n_bins=CONFIG["N_BINS"])
    lgbm_artifact_path = lgbm_calibrated.save(
        f"{CONFIG['OUTPUT_MODEL_DIR']}lgb_churn_model_calibrated.joblib"
    )
    log_message(f"Saved calibrated LightGBM artifact: {lgbm_artifact_path}")

    try:
        log_experiment(
            model_name="LightGBM",
            strategy=f"calibration={lgbm_calibrated.method} (tune_metric={tune_metric})",
            metrics=lgbm_test_results["classification_metrics"],
            features_count=X_train.shape[1],
            threshold=lgbm_calibrated.threshold,
            artifacts_path=str(lgbm_artifact_path),
        )
    except Exception as exc:
        log_message(f"Could not log LightGBM calibration experiment: {exc}", "WARNING")

    banner("SUMMARY")
    cm = lgbm_test_results["classification_metrics"]
    log_message(
        f"LightGBM [{lgbm_calibrated.method}, tuned on '{tune_metric}'] -> "
        f"threshold={lgbm_calibrated.threshold}, "
        f"F1={cm.get('f1')}, ROC-AUC={cm.get('roc_auc')}, PR-AUC={cm.get('pr_auc')}, "
        f"Brier(after)={lgbm_test_results['calibration_after']['brier_score']}, "
        f"ECE(after)={lgbm_test_results['calibration_after']['ece']}"
    )

    return {"LightGBM": lgbm_test_results}


if __name__ == "__main__":
    main()