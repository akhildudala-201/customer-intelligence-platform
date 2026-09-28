
import inspect
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from optuna.samplers import TPESampler
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[1]

for candidate in (str(PROJECT_ROOT), str(APP_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from app.ml.metrics import calculate_metrics, find_optimal_threshold
except ModuleNotFoundError:
    from ml.metrics import calculate_metrics, find_optimal_threshold

try:
    from app.ml.experiment_logger import log_experiment
except ModuleNotFoundError:
    from ml.experiment_logger import log_experiment

# Silence only the known-noisy warnings instead of every warning in the process.
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message=".*has feature names, but.*was fitted without feature names")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Hyperparameters found by an earlier Optuna search (40 trials x 5-fold CV).
# NOTE: they were tuned on an older feature set; set TUNE_HYPERPARAMETERS=True
# to re-tune whenever the feature contract changes.
DEFAULT_BEST_PARAMS = {
    "num_leaves": 23,
    "max_depth": 7,
    "learning_rate": 0.085757,
    "colsample_bytree": 0.807623,
    "subsample": 0.731222,
    "subsample_freq": 6,
    "reg_alpha": 0.224557,
    "reg_lambda": 5.671255,
    "min_child_samples": 53,
}

CHURN_LABEL = 1

CONFIG = {
    "TARGET_COLUMN": "churn_label",
    "ID_COLUMN": "customer_unique_id",
    "RANDOM_STATE": 42,
    # 'weights' (class_weight='balanced'), 'smote', or 'none'
    "RESAMPLER": "weights",
    # Always train on churn (1) as the positive class so P(class 1) == P(churn)
    # everywhere downstream. Set to 0 only for an explicit retention model.
    "POSITIVE_CLASS": CHURN_LABEL,
    "TUNE_HYPERPARAMETERS": False,  # False = use DEFAULT_BEST_PARAMS; True = re-tune with Optuna
    "N_TRIALS": 40,
    "CV_FOLDS": 5,
    "N_ESTIMATORS": 2000,
    "EARLY_STOPPING_ROUNDS": 100,
    # Early stopping / Optuna are scored on PR-AUC of the MINORITY class
    # (retained customers). PR-AUC of a ~97% majority class is ~0.99 for
    # almost any model, so it cannot guide training.
    "EVAL_METRIC": "ap_minority",
    "SMOTE_K_NEIGHBORS": 5,
    # 'score' = fixed probability cut-off tuned on validation (works for any
    # batch size, including single-customer API calls).
    # 'rate'  = flag the top TARGET_FLAG_RATE share of a scored batch
    # (only meaningful for batch scoring).
    "THRESHOLD_MODE": "score",
    "THRESHOLD_METRIC": "balanced_accuracy",
    "TARGET_FLAG_RATE": 0.05,
    "DRIFT_CHECK": True,
    "PSI_SHIFT_THRESHOLD": 0.25,
    "OUTPUT_DIR": str(PROJECT_ROOT / "outputs" / "models") + os.sep,
    "TIMESTAMP": datetime.now().strftime("%Y%m%d_%H%M%S"),
}


os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)


def log_message(message: str, level: str = "INFO") -> None:
    """Format and print application log message with timestamp."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{level}] {message}")


def banner(text: str) -> None:
    """Print standard section banner."""
    log_message("=" * 60)
    log_message(text)
    log_message("=" * 60)


try:
    from app.Database.database import engine
except Exception:
    try:
        from Database.database import engine
    except Exception:
        engine = None

LEAKY_COLS = [
    "single_order_customer",
    "frequency",
    "delivered_orders",
    "canceled_orders",
    "shipped_orders",
    "unavailable_orders",
    "delivered_rate",
    "active_purchase_days",
    "total_items",
    "avg_items_per_order",
    "unique_products",
    "unique_categories",
    "recency_days",
    "tenure_days",
    "review_count",
    "first_purchase_date",
    "last_purchase_date",
    "reference_date",
    "censored",
]


class ChurnLightGBM:
    """LightGBM classifier with class weighting, threshold tuning, and scikit-learn API.

    Used by imbalance_experiments.py. Its API is unchanged.
    """

    def __init__(
        self,
        class_weight: Union[str, Dict[int, float], None] = "balanced",
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        random_state: int = 42,
        threshold: float = 0.5,
        **kwargs: Any,
    ):
        self.class_weight = class_weight
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.random_state = random_state
        self.threshold = threshold
        self.kwargs = kwargs

        self.model = lgb.LGBMClassifier(
            class_weight=self.class_weight,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            random_state=self.random_state,
            objective="binary",
            n_jobs=-1,
            verbose=-1,
            **self.kwargs,
        )
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False
        self.metrics_: Dict[str, Any] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, list],
    ) -> "ChurnLightGBM":
        """Fit the LightGBM classifier on the training data."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            X_arr = np.asarray(X)

        y_arr = np.asarray(y).astype(int)
        if len(np.unique(y_arr)) < 2:
            raise ValueError(f"Training data requires at least two classes. Found: {np.unique(y_arr)}")

        self.model.fit(X_arr, y_arr)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict positive class probabilities."""
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call fit() before predict_proba().")
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
        return self.model.predict_proba(X_arr)[:, 1]

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Predict binary labels using decision threshold."""
        t = self.threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        return (probs >= t).astype(int)

    def tune_threshold(
        self,
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray, list],
        metric: str = "balanced_accuracy",
        thresholds: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Tune decision threshold on validation set to maximize specified metric."""
        probs = self.predict_proba(X_val)
        best_t, best_score = find_optimal_threshold(y_val, probs, metric=metric, thresholds=thresholds)
        self.threshold = best_t
        return best_t, best_score

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, list],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate model on dataset split and return metrics dictionary."""
        t = self.threshold if threshold is None else threshold
        y_probs = self.predict_proba(X)
        y_preds = (y_probs >= t).astype(int)
        metrics = calculate_metrics(y_true=y, y_pred=y_preds, y_prob=y_probs, threshold=t)
        self.metrics_ = metrics
        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances sorted in descending order."""
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call fit() before get_feature_importance().")
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Minority-class metric used for early stopping and Optuna
# ---------------------------------------------------------------------------

def ap_minority_eval(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
    """LightGBM eval metric: PR-AUC of label 0 (retained), higher is better.

    y_pred is P(label == 1), so the score for label 0 is 1 - y_pred.
    Defined at module level so the fitted model stays picklable.
    """
    y_true = np.asarray(y_true).astype(int)
    if y_true.min() == y_true.max():
        return "ap_minority", 0.0, True
    return "ap_minority", float(average_precision_score(1 - y_true, 1 - np.asarray(y_pred))), True


AP_MINORITY_SCORER = make_scorer(
    average_precision_score, response_method="predict_proba", pos_label=0
)


def load_train_val_test():
    """Load train, validation, and test datasets from database."""
    if engine is None:
        raise ConnectionError("Database engine could not be loaded.")
    with engine.connect() as conn:
        train_df = pd.read_sql("SELECT * FROM model_ready_train", conn)
        val_df = pd.read_sql("SELECT * FROM model_ready_val", conn)
        test_df = pd.read_sql("SELECT * FROM model_ready_test", conn)
    return train_df, val_df, test_df


def split_features_target(df: pd.DataFrame):
    """Separate features and target while filtering leaky columns."""
    target = CONFIG["TARGET_COLUMN"]
    id_col = CONFIG["ID_COLUMN"]
    feature_cols = [c for c in df.columns if c not in LEAKY_COLS and c not in [target, id_col]]
    X = df[feature_cols].copy()
    y = df[target].copy()
    return X, y


def load_data():
    """Load dataset splits and log row/column counts.

    Raises instead of calling sys.exit() so callers (run_all.py, tests,
    notebooks) can handle the failure.
    """
    banner("LOADING DATA")
    try:
        train_df, val_df, test_df = load_train_val_test()
    except Exception as exc:
        log_message(f"Failed to load data: {exc}", "ERROR")
        raise RuntimeError(f"Could not load model_ready_* tables: {exc}") from exc

    for name, df in (("Train", train_df), ("Val", val_df), ("Test", test_df)):
        log_message(f"{name}: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return train_df, val_df, test_df


def analyze_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Inspect class distributions and flag potential split prevalence drift."""
    banner("DATA ANALYSIS")
    target = CONFIG["TARGET_COLUMN"]
    prevalence = {}

    for name, df in (("TRAIN", train_df), ("VALIDATION", val_df), ("TEST", test_df)):
        counts = df[target].value_counts().sort_index()
        pct = df[target].value_counts(normalize=True).sort_index() * 100
        log_message(f"{name}: {len(df):,} rows, {df.isnull().sum().sum()} missing values")
        for label in counts.index:
            log_message(f"   label={label}: {counts[label]:,} ({pct[label]:.2f}%)")
        minority = counts.idxmin()
        log_message(
            f"   minority label = {minority} "
            f"({pct[minority]:.2f}%), imbalance {counts.max() / counts.min():.1f}:1"
        )
        prevalence[name] = pct.to_dict()

    labels = sorted(prevalence["TRAIN"].keys())
    for label in labels:
        vals = [prevalence[s].get(label, 0.0) for s in ("TRAIN", "VALIDATION", "TEST")]
        if max(vals) - min(vals) > 1.0:
            log_message(
                f"Prevalence of label={label} drifts across splits "
                f"({vals[0]:.2f}% / {vals[1]:.2f}% / {vals[2]:.2f}%).",
                "WARNING",
            )
    return prevalence


def prepare_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Filter features, remove constant columns, and encode the target class."""
    banner("PREPARING DATA")

    X_train, y_train_raw = split_features_target(train_df)
    X_val, y_val_raw = split_features_target(val_df)
    X_test, y_test_raw = split_features_target(test_df)

    # Drop zero-variance columns in training split
    nunique = X_train.nunique()
    dead = nunique[nunique <= 1].index.tolist()
    if dead:
        log_message(f"Dropping {len(dead)} constant column(s): {dead}", "WARNING")
        X_train = X_train.drop(columns=dead)
        X_val = X_val.drop(columns=dead)
        X_test = X_test.drop(columns=dead)

    X_val = X_val[X_train.columns]
    X_test = X_test[X_train.columns]

    # Positive class is explicit (churn = 1 by default). It is no longer
    # auto-detected as the minority class: that silently flipped the meaning
    # of predict_proba() and SHAP values for every downstream consumer.
    pos = CONFIG.get("POSITIVE_CLASS")
    if pos is None:
        pos = CHURN_LABEL
    pos = int(pos)
    if pos not in (0, 1):
        raise ValueError(f"POSITIVE_CLASS must be 0 or 1, got {pos!r}")
    log_message(
        f"Positive class = original label {pos} "
        f"({'churned' if pos == CHURN_LABEL else 'retained'})"
    )

    y_train = (y_train_raw == pos).astype(int)
    y_val = (y_val_raw == pos).astype(int)
    y_test = (y_test_raw == pos).astype(int)

    log_message(f"Features: {X_train.shape[1]} -> {list(X_train.columns)}")
    log_message(
        f"Train positives: {int(y_train.sum()):,} / {len(y_train):,} "
        f"({100 * y_train.mean():.2f}%)"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, pos


def make_estimator(params: dict, y_train: pd.Series):
    """Instantiate LightGBM classifier with imbalance configuration.

    'weights' uses class_weight='balanced' (n / (2 * n_class)), which is
    symmetric: it gives the same effective up-weighting of the minority class
    whichever label is the positive one.
    """
    params = dict(params)
    params.update(
        n_estimators=params.get("n_estimators", 400),
        objective="binary",
        metric="None",  # the custom minority-class metric is passed at fit time
        random_state=CONFIG["RANDOM_STATE"],
        n_jobs=-1,
        verbose=-1,
    )

    if CONFIG["RESAMPLER"] == "weights":
        params["class_weight"] = "balanced"

    clf = lgb.LGBMClassifier(**params)

    if CONFIG["RESAMPLER"] == "smote":
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline

        return Pipeline(
            [
                (
                    "smote",
                    SMOTE(
                        k_neighbors=CONFIG["SMOTE_K_NEIGHBORS"],
                        random_state=CONFIG["RANDOM_STATE"],
                    ),
                ),
                ("clf", clf),
            ]
        )
    return clf


def objective_function(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """Optuna objective: stratified CV PR-AUC of the minority (retained) class."""
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 8, 48),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
        "subsample": trial.suggest_float("subsample", 0.5, 0.95),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 500),
        "n_estimators": 400,
    }
    estimator = make_estimator(params, y_train)
    cv = StratifiedKFold(
        n_splits=CONFIG["CV_FOLDS"], shuffle=True, random_state=CONFIG["RANDOM_STATE"]
    )
    scores = cross_val_score(
        estimator, X_train, y_train, cv=cv, scoring=AP_MINORITY_SCORER, n_jobs=-1
    )
    return float(scores.mean())


def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Return hyperparameters (cached defaults, or a fresh Optuna search)."""
    banner("HYPERPARAMETER CONFIGURATION")

    if not CONFIG.get("TUNE_HYPERPARAMETERS", False):
        log_message(
            "Using cached DEFAULT_BEST_PARAMS (no tuning this run; set "
            "CONFIG['TUNE_HYPERPARAMETERS']=True to re-tune with Optuna)"
        )
        for k, v in DEFAULT_BEST_PARAMS.items():
            log_message(f"   {k}: {v}")
        return dict(DEFAULT_BEST_PARAMS)

    baseline = float(1 - y_train.mean())
    log_message(f"Random-guess minority PR-AUC baseline = {baseline:.4f}")
    log_message(f"{CONFIG['N_TRIALS']} trials x {CONFIG['CV_FOLDS']}-fold CV")

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=CONFIG["RANDOM_STATE"])
    )
    study.optimize(
        lambda t: objective_function(t, X_train, y_train),
        n_trials=CONFIG["N_TRIALS"],
        show_progress_bar=False,
    )

    log_message(f"Best CV minority PR-AUC: {study.best_value:.4f} (vs {baseline:.4f} baseline)")
    for k, v in study.best_params.items():
        log_message(f"   {k}: {v}")
    return dict(study.best_params)


def _eval_data_kwargs(X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Validation-data arguments for LGBMClassifier.fit().

    LightGBM >= 4.7 renamed eval_set=[(X, y)] to eval_X=(X,), eval_y=(y,)
    and warns on the old name; older versions only accept eval_set.
    """
    if "eval_X" in inspect.signature(lgb.LGBMClassifier.fit).parameters:
        return {"eval_X": (X_val,), "eval_y": (y_val,)}
    return {"eval_set": [(X_val, y_val)]}


def train_final_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, best_params: dict):
    """Fit final LightGBM model with early stopping on the validation split.

    Note: the validation split is also used to choose the decision threshold
    (and later for calibration), so validation metrics are optimistic.
    Only TEST metrics should be reported.
    """
    banner("TRAINING FINAL MODEL")

    params = dict(best_params)
    params["n_estimators"] = CONFIG["N_ESTIMATORS"]
    estimator = make_estimator(params, y_train)

    callbacks = [
        lgb.early_stopping(CONFIG["EARLY_STOPPING_ROUNDS"], first_metric_only=True, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    fit_kwargs = {**_eval_data_kwargs(X_val, y_val), "eval_metric": ap_minority_eval, "callbacks": callbacks}

    if CONFIG["RESAMPLER"] == "smote":
        estimator.fit(X_train, y_train, **{f"clf__{k}": v for k, v in fit_kwargs.items()})
        booster = estimator.named_steps["clf"]
    else:
        estimator.fit(X_train, y_train, **fit_kwargs)
        booster = estimator

    log_message(f"Trees kept by early stopping: {booster.best_iteration_}")
    log_message(f"Best validation score: {dict(booster.best_score_)}")
    return estimator


def choose_operating_point(model, X_val: pd.DataFrame, y_val: pd.Series):
    """Choose the decision rule on validation.

    SCORE mode (default): probability cut-off that maximises
    CONFIG['THRESHOLD_METRIC'] (balanced accuracy by default, the same metric
    model_comparison.py and calibration.py tune on).
    RATE mode: flag the top TARGET_FLAG_RATE share of each scored batch.
    """
    banner("CHOOSING OPERATING POINT")
    proba = model.predict_proba(X_val)[:, 1]
    y_val = np.asarray(y_val).astype(int)

    metric = CONFIG.get("THRESHOLD_METRIC", "balanced_accuracy")
    best_t, best_score = find_optimal_threshold(y_val, proba, metric=metric)
    best_t = float(best_t)
    pred = (proba >= best_t).astype(int)
    log_message(
        f"Best {metric} threshold on validation: {best_t:.4f} ({metric}={best_score:.4f}, "
        f"flags {100 * pred.mean():.1f}% of rows as positive)"
    )
    log_message(
        f"Fixed cut-off 0.5 -> balanced accuracy "
        f"{balanced_accuracy_score(y_val, proba >= 0.5):.4f}"
    )

    # Customers ranked as LEAST likely to be positive (i.e. most likely to be
    # retained when positive = churn). These are the useful targeting lists.
    minority = 0 if y_val.mean() >= 0.5 else 1
    score_for_minority = (1 - proba) if minority == 0 else proba
    is_minority = (y_val == minority).astype(int)
    base = max(is_minority.mean(), 1e-9)
    log_message(f"Rate-based lists for the minority label ({minority}) on validation:")
    for k_pct in (1, 2, 5, 10, 20, 30):
        k = max(int(len(proba) * k_pct / 100), 1)
        idx = np.argsort(-score_for_minority, kind="stable")[:k]
        prec = float(np.mean(is_minority[idx]))
        rec = float(np.sum(is_minority[idx]) / max(np.sum(is_minority), 1))
        log_message(
            f"   top {k_pct:>2}%: precision {prec:.4f}, recall {rec:.4f}, "
            f"lift {prec / base:.1f}x"
        )

    if CONFIG["THRESHOLD_MODE"] == "rate":
        rate = float(CONFIG["TARGET_FLAG_RATE"])
        log_message(
            f"Using RATE mode: flag the top {100 * rate:.0f}% of each scored batch. "
            "Not suitable for single-customer scoring.",
            "WARNING",
        )
        return ("rate", rate)

    log_message(f"Using SCORE mode: fixed cut-off {best_t:.4f} on P(class 1).")
    return ("score", best_t)


def apply_operating_point(proba: np.ndarray, mode: str, value: float) -> np.ndarray:
    """Convert predicted probability scores to binary classification decisions.

    rate mode flags exactly round(n * value) rows (ties broken by position),
    and flags nothing when that rounds to 0 (e.g. a single customer).
    """
    proba = np.asarray(proba, dtype=float)
    if mode == "rate":
        k = int(round(len(proba) * value))
        pred = np.zeros(len(proba), dtype=int)
        if k <= 0:
            return pred
        top = np.argsort(-proba, kind="stable")[:k]
        pred[top] = 1
        return pred
    return (proba >= value).astype(int)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a reference and a new distribution.

    - Discrete features (<= `bins` distinct reference values, e.g. 0/1 flags)
      are compared category by category.
    - Continuous features use reference-quantile bins with open outer edges
      (-inf, +inf), so values outside the training range are still counted.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e = e[~np.isnan(e)]
    a = a[~np.isnan(a)]
    if len(e) == 0 or len(a) == 0:
        return 0.0

    ref_values = np.unique(e)
    if len(ref_values) <= bins:
        categories = np.union1d(ref_values, np.unique(a))
        e_pct = np.array([np.mean(e == c) for c in categories])
        a_pct = np.array([np.mean(a == c) for c in categories])
    else:
        inner = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)[1:-1]))
        edges = np.concatenate(([-np.inf], inner, [np.inf]))
        e_pct = np.histogram(e, bins=edges)[0] / len(e)
        a_pct = np.histogram(a, bins=edges)[0] / len(a)

    e_pct = np.clip(e_pct, 1e-6, None)
    a_pct = np.clip(a_pct, 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def drift_report(model, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """Evaluate feature and score distribution stability across splits."""
    banner("DRIFT REPORT (train vs val vs test)")
    limit = CONFIG.get("PSI_SHIFT_THRESHOLD", 0.25)
    rows = []
    for col in X_train.columns:
        rows.append(
            {
                "feature": col,
                "psi_val": psi(X_train[col].values, X_val[col].values),
                "psi_test": psi(X_train[col].values, X_test[col].values),
                "mean_train": X_train[col].mean(),
                "mean_val": X_val[col].mean(),
                "mean_test": X_test[col].mean(),
            }
        )
    df = pd.DataFrame(rows).sort_values("psi_test", ascending=False)
    for _, r in df.iterrows():
        flag = "  <-- SHIFTED" if r["psi_test"] > limit else ""
        log_message(
            f"   {r['feature']}: PSI val {r['psi_val']:.3f}, test {r['psi_test']:.3f} | "
            f"mean {r['mean_train']:.2f} / {r['mean_val']:.2f} / {r['mean_test']:.2f}{flag}"
        )

    p_tr = model.predict_proba(X_train)[:, 1]
    p_v = model.predict_proba(X_val)[:, 1]
    p_te = model.predict_proba(X_test)[:, 1]
    log_message(f"Score PSI: val {psi(p_tr, p_v):.3f}, test {psi(p_tr, p_te):.3f}")
    log_message("Predicted-score distribution (mean / 5th pct / 95th pct):")
    for name, p in (("train", p_tr), ("val", p_v), ("test", p_te)):
        log_message(
            f"   {name:<5} {p.mean():.4f} / {np.quantile(p, 0.05):.4f} / "
            f"{np.quantile(p, 0.95):.4f}"
        )
    return df


def evaluate_model(model, mode: str, value: float, splits: list) -> dict:
    """Evaluate model metrics for both classes on all splits.

    With churn as the positive class (~95% of rows), churn precision / PR-AUC
    sit close to their baselines for any model, so balanced accuracy and the
    minority-class (retained) metrics are reported alongside them.
    """
    banner(f"MODEL EVALUATION (operating point: {mode} = {value:.4f})")
    results = {}

    for name, X, y in splits:
        y = np.asarray(y).astype(int)
        proba = model.predict_proba(X)[:, 1]
        pred = apply_operating_point(proba, mode, value)

        metrics = {
            "prevalence": float(np.mean(y)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, proba)),
            "pr_auc": float(average_precision_score(y, proba)),
            # Class-0 view (retained customers when positive = churn)
            "neg_prevalence": float(1 - np.mean(y)),
            "neg_precision": float(precision_score(1 - y, 1 - pred, zero_division=0)),
            "neg_recall": float(recall_score(1 - y, 1 - pred, zero_division=0)),
            "neg_pr_auc": float(average_precision_score(1 - y, 1 - proba)),
            "lift": 0.0,
        }
        metrics["lift"] = (
            metrics["precision"] / metrics["prevalence"] if metrics["prevalence"] else 0.0
        )

        log_message(f"{name}:")
        log_message(f"   class 1 share     {int(np.sum(y)):,} / {len(y):,} ({100 * metrics['prevalence']:.2f}%)")
        log_message(f"   ROC-AUC           {metrics['roc_auc']:.4f}")
        log_message(f"   balanced accuracy {metrics['balanced_accuracy']:.4f}")
        log_message(f"   class 1: PR-AUC {metrics['pr_auc']:.4f} (baseline {metrics['prevalence']:.4f}), "
                    f"precision {metrics['precision']:.4f}, recall {metrics['recall']:.4f}")
        log_message(f"   class 0: PR-AUC {metrics['neg_pr_auc']:.4f} (baseline {metrics['neg_prevalence']:.4f}), "
                    f"precision {metrics['neg_precision']:.4f}, recall {metrics['neg_recall']:.4f}")
        log_message(f"   flagged class 1   {int(pred.sum()):,} rows ({100 * pred.mean():.1f}%)")
        log_message(f"   confusion matrix [[tn fp],[fn tp]] = {confusion_matrix(y, pred, labels=[0, 1]).tolist()}")

        metrics.update(y_true=y, y_pred=pred, y_proba=proba)
        results[name] = metrics

    return results


def plot_feature_importance(model, feature_cols: list, top_n: int = 20) -> pd.DataFrame:
    """Generate and save feature importance plot based on split gain."""
    banner("FEATURE IMPORTANCE (gain)")
    booster = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    gain = booster.booster_.feature_importance(importance_type="gain")
    split = booster.booster_.feature_importance(importance_type="split")

    imp = (
        pd.DataFrame({"feature": feature_cols, "gain": gain, "splits": split})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    imp["gain_pct"] = 100 * imp["gain"] / max(imp["gain"].sum(), 1e-9)
    for _, row in imp.head(top_n).iterrows():
        log_message(f"   {row['feature']}: gain {row['gain']:.1f} ({row['gain_pct']:.1f}%), splits {int(row['splits'])}")

    unused = imp[imp["splits"] == 0]["feature"].tolist()
    if unused:
        log_message(f"Never used by any split: {unused}", "WARNING")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp.head(top_n), x="gain", y="feature", color="steelblue")
    plt.title("Feature importance (gain)")
    plt.tight_layout()
    path = f"{CONFIG['OUTPUT_DIR']}feature_importance_{CONFIG['TIMESTAMP']}.png"
    plt.savefig(path, dpi=100)
    plt.close()
    log_message(f"Saved {path}")
    return imp


def create_visualizations(results: dict) -> None:
    """Plot confusion matrices, ROC curves, and Precision-Recall curves."""
    banner("VISUALISATIONS")
    ts = CONFIG["TIMESTAMP"]

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    axes = np.atleast_1d(axes)
    for ax, (name, data) in zip(axes, results.items()):
        sns.heatmap(
            confusion_matrix(data["y_true"], data["y_pred"], labels=[0, 1]),
            annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{CONFIG['OUTPUT_DIR']}confusion_matrices_{ts}.png", dpi=100)
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
        ax1.plot(fpr, tpr, label=f"{name} (AUC={data['roc_auc']:.3f})", linewidth=2)
        # PR curve for the minority class (class 0), where it is informative.
        prec, rec, _ = precision_recall_curve(1 - data["y_true"], 1 - data["y_proba"])
        ax2.plot(rec, prec, label=f"{name} (AP={data['neg_pr_auc']:.3f})", linewidth=2)
        ax2.axhline(data["neg_prevalence"], linestyle=":", linewidth=1)
    ax1.plot([0, 1], [0, 1], "k--", label="random")
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.set_title("ROC")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("Recall (class 0)")
    ax2.set_ylabel("Precision (class 0)")
    ax2.set_title("Precision-Recall, class 0 (dotted = baseline)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['OUTPUT_DIR']}curves_{ts}.png", dpi=100)
    plt.close()
    log_message(f"Saved plots to {CONFIG['OUTPUT_DIR']}")


def _portable_config() -> dict:
    """CONFIG copy with OUTPUT_DIR stored relative to the project root, so
    metadata does not contain a developer's absolute (e.g. Windows) path."""
    cfg = {k: v for k, v in CONFIG.items()}
    try:
        cfg["OUTPUT_DIR"] = Path(os.path.relpath(CONFIG["OUTPUT_DIR"], PROJECT_ROOT)).as_posix() + "/"
    except ValueError:  # different drive on Windows
        cfg["OUTPUT_DIR"] = "outputs/models/"
    return cfg


def save_model(model, feature_cols: list, results: dict, mode: str, value: float, positive_class: int, best_params: dict):
    """Serialize model artifact and metadata to disk.

    Artifact keys are unchanged, so calibration.py, model_comparison.py,
    threshold_analysis.py and the inference API keep working.
    """
    banner("SAVING MODEL")
    ts = CONFIG["TIMESTAMP"]
    model_path = f"{CONFIG['OUTPUT_DIR']}lgb_churn_model_{ts}.joblib"

    artifact = {
        "model": model,
        "feature_cols": list(feature_cols),
        "operating_point": {"mode": mode, "value": value},
        "positive_class": positive_class,
        "probability_definition": f"P(churn_label == {positive_class})",
        "config": _portable_config(),
        "best_params": best_params,
        "metrics": {
            name: {k: v for k, v in m.items() if not isinstance(v, np.ndarray)}
            for name, m in results.items()
        },
    }
    joblib.dump(artifact, model_path)
    log_message(f"Saved {model_path}")

    meta_path = f"{CONFIG['OUTPUT_DIR']}metadata_{ts}.json"
    with open(meta_path, "w") as fh:
        json.dump({k: v for k, v in artifact.items() if k != "model"}, fh, indent=2, default=str)
    log_message(f"Saved {meta_path}")
    return model_path, meta_path


def _strategy_label() -> str:
    """Describe this run truthfully in the experiment log."""
    weighting = {
        "weights": "class_weight=balanced",
        "smote": "SMOTE",
        "none": "no reweighting",
    }.get(CONFIG["RESAMPLER"], CONFIG["RESAMPLER"])
    tuning = (
        f"Optuna {CONFIG['N_TRIALS']} trials"
        if CONFIG.get("TUNE_HYPERPARAMETERS")
        else "cached params, not re-tuned"
    )
    return f"{weighting}; {tuning}; early stop on {CONFIG['EVAL_METRIC']}"


def main():
    """Execute end-to-end LightGBM churn classification pipeline."""
    banner("LIGHTGBM CHURN PREDICTION PIPELINE")

    train_df, val_df, test_df = load_data()
    analyze_data(train_df, val_df, test_df)

    X_train, y_train, X_val, y_val, X_test, y_test, positive_class = prepare_data(
        train_df, val_df, test_df
    )

    best_params = hyperparameter_tuning(X_train, y_train)
    model = train_final_model(X_train, y_train, X_val, y_val, best_params)

    if CONFIG["DRIFT_CHECK"]:
        drift_report(model, X_train, X_val, X_test)

    mode, value = choose_operating_point(model, X_val, y_val)

    results = evaluate_model(
        model,
        mode,
        value,
        [("TRAIN", X_train, y_train), ("VALIDATION", X_val, y_val), ("TEST", X_test, y_test)],
    )

    plot_feature_importance(model, X_train.columns.tolist())
    create_visualizations(results)
    model_path, meta_path = save_model(model, X_train.columns, results, mode, value, positive_class, best_params)

    test = results["TEST"]
    try:
        run_id = log_experiment(
            model_name="LightGBM",
            strategy=_strategy_label(),
            metrics=test,
            features_count=len(X_train.columns),
            threshold=f"{mode}={value:.4f}",
            artifacts_path=Path(os.path.relpath(model_path, PROJECT_ROOT)).as_posix(),
        )
        log_message(f"Logged run #{run_id} to outputs/reports/experiment_log.csv")
    except Exception as exc:
        log_message(f"Could not log experiment: {exc}", "WARNING")

    banner("SUMMARY (test split)")
    log_message(f"ROC-AUC            {test['roc_auc']:.4f}")
    log_message(f"Balanced accuracy  {test['balanced_accuracy']:.4f}")
    log_message(
        f"Class {positive_class} PR-AUC     {test['pr_auc']:.4f} (baseline {test['prevalence']:.4f})"
    )
    log_message(
        f"Class {1 - positive_class} PR-AUC     {test['neg_pr_auc']:.4f} (baseline {test['neg_prevalence']:.4f}) "
        f"= {test['neg_pr_auc'] / max(test['neg_prevalence'], 1e-9):.1f}x baseline"
    )
    log_message(
        f"Operating point {mode}={value:.4f}: class {positive_class} precision {test['precision']:.4f}, "
        f"recall {test['recall']:.4f} | class {1 - positive_class} precision {test['neg_precision']:.4f}, "
        f"recall {test['neg_recall']:.4f}"
    )
    log_message(f"predict_proba()[:, 1] = P(churn_label == {positive_class}).")


if __name__ == "__main__":
    main()