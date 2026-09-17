"""LightGBM training pipeline with Optuna hyperparameter optimization and drift evaluation."""

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
    classification_report,
    confusion_matrix,
    f1_score,
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

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Optimal hyperparameters discovered by Optuna (40 trials x 5-fold CV on PR-AUC)
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

CONFIG = {
    "TARGET_COLUMN": "churn_label",
    "ID_COLUMN": "customer_unique_id",
    "RANDOM_STATE": 42,
    "RESAMPLER": "weights",  # 'weights' (scale_pos_weight) or 'smote' or 'none'
    "POSITIVE_CLASS": None,  # None auto-detects minority class
    "TUNE_HYPERPARAMETERS": False,  # False = instant 3-second run using DEFAULT_BEST_PARAMS; True = re-tune with Optuna
    "N_TRIALS": 40,
    "CV_FOLDS": 5,
    "N_ESTIMATORS": 2000,
    "EARLY_STOPPING_ROUNDS": 100,
    "EVAL_METRIC": "average_precision",
    "SMOTE_K_NEIGHBORS": 5,
    "THRESHOLD_MODE": "rate",  # 'rate' (top population slice) or 'score' (probability cutoff)
    "TARGET_FLAG_RATE": 0.05,
    "DRIFT_CHECK": True,
    "OUTPUT_DIR": str(PROJECT_ROOT / "outputs" / "models") + "/",
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
    """LightGBM classifier with class weighting, threshold tuning, and scikit-learn API."""

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


HAS_FEATURE_PREP = True


def load_data():
    """Load dataset splits and log row/column counts."""
    banner("LOADING DATA")
    if load_train_val_test is None:
        log_message("load_train_val_test function not found.", "ERROR")
        sys.exit(1)
    try:
        train_df, val_df, test_df = load_train_val_test()
    except Exception as exc:
        log_message(f"Failed to load data: {exc}", "ERROR")
        sys.exit(1)

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

    if HAS_FEATURE_PREP:
        X_train, y_train_raw = split_features_target(train_df)
        X_val, y_val_raw = split_features_target(val_df)
        X_test, y_test_raw = split_features_target(test_df)
    else:
        exclude = [CONFIG["TARGET_COLUMN"], CONFIG["ID_COLUMN"]]
        cols = [c for c in train_df.columns if c not in exclude]
        X_train, y_train_raw = train_df[cols].copy(), train_df[CONFIG["TARGET_COLUMN"]]
        X_val, y_val_raw = val_df[cols].copy(), val_df[CONFIG["TARGET_COLUMN"]]
        X_test, y_test_raw = test_df[cols].copy(), test_df[CONFIG["TARGET_COLUMN"]]

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

    # Map minority class of interest to 1
    pos = CONFIG["POSITIVE_CLASS"]
    if pos is None:
        pos = int(y_train_raw.value_counts().idxmin())
        log_message(f"Auto-selected positive class = original label {pos} (minority)")
    else:
        log_message(f"Positive class = original label {pos} (from CONFIG)")

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
    """Instantiate LightGBM classifier with imbalance configuration."""
    params = dict(params)
    params.update(
        n_estimators=params.get("n_estimators", 400),
        objective="binary",
        metric=CONFIG["EVAL_METRIC"],
        random_state=CONFIG["RANDOM_STATE"],
        n_jobs=-1,
        verbose=-1,
    )

    if CONFIG["RESAMPLER"] == "weights":
        pos = int(np.sum(y_train == 1))
        neg = int(np.sum(y_train == 0))
        params["scale_pos_weight"] = neg / max(pos, 1)

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
    """Optuna objective function maximizing stratified cross-validation PR-AUC."""
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
        estimator, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1
    )
    return float(scores.mean())


def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Return optimal hyperparameters (using cached best_params or running Optuna search)."""
    banner("HYPERPARAMETER CONFIGURATION")

    if not CONFIG.get("TUNE_HYPERPARAMETERS", False):
        log_message("Using pre-tuned optimal hyperparameters (set CONFIG['TUNE_HYPERPARAMETERS']=True to re-tune with Optuna)")
        for k, v in DEFAULT_BEST_PARAMS.items():
            log_message(f"   {k}: {v}")
        return dict(DEFAULT_BEST_PARAMS)

    baseline = float(y_train.mean())
    log_message(f"Random-guess PR-AUC baseline = prevalence = {baseline:.4f}")
    log_message(f"{CONFIG['N_TRIALS']} trials x {CONFIG['CV_FOLDS']}-fold CV")

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=CONFIG["RANDOM_STATE"])
    )
    study.optimize(
        lambda t: objective_function(t, X_train, y_train),
        n_trials=CONFIG["N_TRIALS"],
        show_progress_bar=False,
    )

    log_message(f"Best CV PR-AUC: {study.best_value:.4f} (vs {baseline:.4f} baseline)")
    for k, v in study.best_params.items():
        log_message(f"   {k}: {v}")
    return dict(study.best_params)


def train_final_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, best_params: dict):
    """Fit final LightGBM model with early stopping on validation split."""
    banner("TRAINING FINAL MODEL")

    params = dict(best_params)
    params["n_estimators"] = CONFIG["N_ESTIMATORS"]
    estimator = make_estimator(params, y_train)

    callbacks = [
        lgb.early_stopping(CONFIG["EARLY_STOPPING_ROUNDS"], first_metric_only=True, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    if CONFIG["RESAMPLER"] == "smote":
        estimator.fit(
            X_train,
            y_train,
            clf__eval_set=[(X_val, y_val)],
            clf__eval_metric=CONFIG["EVAL_METRIC"],
            clf__callbacks=callbacks,
        )
        booster = estimator.named_steps["clf"]
    else:
        estimator.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=CONFIG["EVAL_METRIC"],
            callbacks=callbacks,
        )
        booster = estimator

    log_message(f"Trees kept by early stopping: {booster.best_iteration_}")
    log_message(f"Best validation score: {booster.best_score_}")
    return estimator


def choose_operating_point(model, X_val: pd.DataFrame, y_val: pd.Series):
    """Determine probability decision threshold or percentile-based operating rate."""
    banner("CHOOSING OPERATING POINT")
    proba = model.predict_proba(X_val)[:, 1]
    y_val = np.asarray(y_val)

    precision, recall, thresholds = precision_recall_curve(y_val, proba)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    best = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    f1_threshold = float(thresholds[best]) if len(thresholds) else 0.5

    log_message(f"Fixed cut-off 0.5      -> F1 {f1_score(y_val, proba >= 0.5, zero_division=0):.4f}")
    log_message(
        f"Fixed cut-off {f1_threshold:.4f} -> F1 {f1[best]:.4f} "
        f"(precision {precision[best]:.4f}, recall {recall[best]:.4f}, "
        f"flags {100 * np.mean(proba >= f1_threshold):.1f}% of rows)"
    )

    log_message("Rate-based operating points on validation:")
    for k_pct in (1, 2, 5, 10, 20, 30):
        k = max(int(len(proba) * k_pct / 100), 1)
        idx = np.argsort(-proba)[:k]
        prec = float(np.mean(y_val[idx]))
        rec = float(np.sum(y_val[idx]) / max(np.sum(y_val), 1))
        f1k = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        log_message(
            f"   top {k_pct:>2}%: precision {prec:.4f}, recall {rec:.4f}, "
            f"F1 {f1k:.4f}, lift {prec / max(y_val.mean(), 1e-9):.1f}x"
        )

    if CONFIG["THRESHOLD_MODE"] == "rate":
        rate = CONFIG["TARGET_FLAG_RATE"]
        log_message(f"Using RATE mode: flag the top {100 * rate:.0f}% of each scored population.")
        return ("rate", float(rate))

    log_message(f"Using SCORE mode: fixed cut-off {f1_threshold:.4f}.", "WARNING")
    return ("score", f1_threshold)


def apply_operating_point(proba: np.ndarray, mode: str, value: float) -> np.ndarray:
    """Convert predicted probability scores to binary classification decisions."""
    if mode == "rate":
        k = max(int(round(len(proba) * value)), 1)
        cut = np.sort(proba)[::-1][k - 1]
        return (proba >= cut).astype(int)
    return (proba >= value).astype(int)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index between two distributions."""
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    e = np.histogram(expected, bins=cuts)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=cuts)[0] / max(len(actual), 1)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(model, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """Evaluate feature and score distribution stability across splits."""
    banner("DRIFT REPORT (train vs val vs test)")
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
        flag = "  <-- SHIFTED" if r["psi_test"] > 0.25 else ""
        log_message(
            f"   {r['feature']}: PSI val {r['psi_val']:.3f}, test {r['psi_test']:.3f} | "
            f"mean {r['mean_train']:.2f} / {r['mean_val']:.2f} / {r['mean_test']:.2f}{flag}"
        )

    p_tr = model.predict_proba(X_train)[:, 1]
    p_v = model.predict_proba(X_val)[:, 1]
    p_te = model.predict_proba(X_test)[:, 1]
    log_message("Predicted-score distribution (mean / 95th pct / share above 0.9):")
    for name, p in (("train", p_tr), ("val", p_v), ("test", p_te)):
        log_message(
            f"   {name:<5} {p.mean():.4f} / {np.quantile(p, 0.95):.4f} / "
            f"{np.mean(p > 0.9):.4f}"
        )
    return df


def evaluate_model(model, mode: str, value: float, splits: list) -> dict:
    """Evaluate model metrics, confusion matrix, and classification report on all splits."""
    banner(f"MODEL EVALUATION (operating point: {mode} = {value:.4f})")
    results = {}

    for name, X, y in splits:
        proba = model.predict_proba(X)[:, 1]
        pred = apply_operating_point(proba, mode, value)

        metrics = {
            "prevalence": float(np.mean(y)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, proba)),
            "pr_auc": float(average_precision_score(y, proba)),
            "lift": 0.0,
        }
        metrics["lift"] = (
            metrics["precision"] / metrics["prevalence"] if metrics["prevalence"] else 0.0
        )

        log_message(f"{name}:")
        log_message(f"   positives      {int(np.sum(y)):,} / {len(y):,} ({100 * metrics['prevalence']:.2f}%)")
        log_message(f"   PR-AUC         {metrics['pr_auc']:.4f}  (baseline {metrics['prevalence']:.4f})")
        log_message(f"   ROC-AUC        {metrics['roc_auc']:.4f}")
        log_message(f"   precision      {metrics['precision']:.4f}")
        log_message(f"   recall         {metrics['recall']:.4f}")
        log_message(f"   F1             {metrics['f1']:.4f}")
        log_message(f"   lift over base {metrics['lift']:.2f}x")
        log_message(f"   flagged        {int(pred.sum()):,} rows ({100 * pred.mean():.1f}%)")
        log_message(f"   confusion matrix [[tn fp],[fn tp]] = {confusion_matrix(y, pred).tolist()}")

        metrics.update(y_true=np.asarray(y), y_pred=pred, y_proba=proba)
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
            confusion_matrix(data["y_true"], data["y_pred"]),
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
        prec, rec, _ = precision_recall_curve(data["y_true"], data["y_proba"])
        ax2.plot(rec, prec, label=f"{name} (AP={data['pr_auc']:.3f})", linewidth=2)
        ax2.axhline(data["prevalence"], linestyle=":", linewidth=1)
    ax1.plot([0, 1], [0, 1], "k--", label="random")
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.set_title("ROC")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['OUTPUT_DIR']}curves_{ts}.png", dpi=100)
    plt.close()
    log_message(f"Saved plots to {CONFIG['OUTPUT_DIR']}")


def save_model(model, feature_cols: list, results: dict, mode: str, value: float, positive_class: int, best_params: dict):
    """Serialize model artifact and metadata to disk."""
    banner("SAVING MODEL")
    ts = CONFIG["TIMESTAMP"]
    model_path = f"{CONFIG['OUTPUT_DIR']}lgb_churn_model_{ts}.joblib"

    artifact = {
        "model": model,
        "feature_cols": list(feature_cols),
        "operating_point": {"mode": mode, "value": value},
        "positive_class": positive_class,
        "config": {k: v for k, v in CONFIG.items()},
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
            strategy=f"scale_pos_weight (Optuna {CONFIG['N_TRIALS']} trials)",
            metrics=test,
            features_count=len(X_train.columns),
            threshold=f"{mode}={value:.4f}",
            artifacts_path=model_path,
        )
        log_message(f"Logged run #{run_id} to outputs/reports/experiment_log.csv")
    except Exception as exc:
        log_message(f"Could not log experiment: {exc}", "WARNING")

    banner("SUMMARY")
    log_message(
        f"Test PR-AUC  {test['pr_auc']:.4f} (baseline {test['prevalence']:.4f}) "
        f"= {test['pr_auc'] / max(test['prevalence'], 1e-9):.1f}x baseline"
    )
    log_message(f"Test ROC-AUC {test['roc_auc']:.4f}")
    log_message(
        f"Operating point {mode}={value:.4f}: precision {test['precision']:.4f}, "
        f"recall {test['recall']:.4f}, lift {test['lift']:.2f}x"
    )
    log_message(f"Reported metrics are for original label {positive_class}.")


if __name__ == "__main__":
    main()