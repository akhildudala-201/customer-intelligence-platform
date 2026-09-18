"""Logistic Regression model pipeline for customer churn classification."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

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
    from app.Database.database import engine
except (ModuleNotFoundError, ImportError):
    engine = None

NON_FEATURE_COLUMNS = [
    "customer_unique_id",
    "churn_label",
    "censored",
    "first_purchase_date",
    "last_purchase_date",
    "reference_date",
    # Excluded leakage features and lifetime order counts
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
]
LABEL_COLUMN = "churn_label"


class ChurnLogisticRegression:
    """Logistic Regression classifier with class weighting and threshold tuning."""

    def __init__(
        self,
        class_weight: Union[str, Dict[int, float], None] = "balanced",
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
        threshold: float = 0.5,
    ):
        self.class_weight = class_weight
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.threshold = threshold

        sklearn_kwargs: Dict[str, Any] = {
            "class_weight": self.class_weight,
            "C": self.C,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }
        if self.penalty not in ("l2", "deprecated", None):
            sklearn_kwargs["penalty"] = self.penalty

        self.model = LogisticRegression(**sklearn_kwargs)

        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False
        self.metrics_: Dict[str, Any] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, list],
    ) -> "ChurnLogisticRegression":
        """
        Fit the logistic regression model according to the given training data.
        """
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
        """Predict probabilities for the positive class (churn=1)."""
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
        """Predict binary class labels using the decision threshold."""
        t = self.threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        return (probs >= t).astype(int)

    def tune_threshold(
        self,
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray, list],
        metric: str = "f1",
        thresholds: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Tune decision threshold on validation data to maximize the target metric."""
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
        """Evaluate predictions on a dataset split and return metrics dictionary."""
        t = self.threshold if threshold is None else threshold
        y_probs = self.predict_proba(X)
        y_preds = (y_probs >= t).astype(int)

        metrics = calculate_metrics(y_true=y, y_pred=y_preds, y_prob=y_probs, threshold=t)
        self.metrics_ = metrics
        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Compute feature coefficients and odds ratios (exp(coef))."""
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call fit() before get_feature_importance().")

        coefs = self.model.coef_[0]
        odds_ratios = np.exp(coefs)

        df_importance = pd.DataFrame({
            "feature": self.feature_names_,
            "coefficient": np.round(coefs, 4),
            "odds_ratio": np.round(odds_ratios, 4),
            "abs_coefficient": np.round(np.abs(coefs), 4),
        }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

        return df_importance

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize model artifact and metadata to disk."""
        if not self.is_fitted_:
            raise ValueError("Cannot save an unfitted model.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model": self.model,
            "feature_names_": self.feature_names_,
            "threshold": self.threshold,
            "class_weight": self.class_weight,
            "C": self.C,
            "penalty": self.penalty,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "metrics_": self.metrics_,
        }
        joblib.dump(bundle, path)
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ChurnLogisticRegression":
        """Load serialized model artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")

        bundle = joblib.load(path)
        instance = cls(
            class_weight=bundle.get("class_weight", "balanced"),
            C=bundle.get("C", 1.0),
            penalty=bundle.get("penalty", "l2"),
            solver=bundle.get("solver", "lbfgs"),
            max_iter=bundle.get("max_iter", 1000),
            random_state=bundle.get("random_state", 42),
            threshold=bundle.get("threshold", 0.5),
        )
        instance.model = bundle["model"]
        instance.feature_names_ = bundle.get("feature_names_", [])
        instance.metrics_ = bundle.get("metrics_", {})
        instance.is_fitted_ = True
        return instance


def separate_features_and_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into feature matrix X and target series y."""
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Target column '{LABEL_COLUMN}' not found in dataframe.")

    X = df[feature_cols].copy()
    y = df[LABEL_COLUMN].astype(int)
    return X, y


def load_model_ready_data(
    con=None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train, validation, and test tables from database."""
    connection = con or engine
    if connection is None:
        raise ConnectionError("No valid database engine found to load model_ready tables.")

    train_df = pd.read_sql("SELECT * FROM model_ready_train", con=connection)
    val_df = pd.read_sql("SELECT * FROM model_ready_val", con=connection)
    test_df = pd.read_sql("SELECT * FROM model_ready_test", con=connection)

    X_train, y_train = separate_features_and_label(train_df)
    X_val, y_val = separate_features_and_label(val_df)
    X_test, y_test = separate_features_and_label(test_df)

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_weight: Union[str, Dict[int, float], None] = "balanced",
    tune_threshold: bool = True,
    tune_metric: str = "balanced_accuracy",
    save_path: Optional[Union[str, Path]] = None,
    log_run: bool = True,
) -> Tuple[ChurnLogisticRegression, Dict[str, Any]]:
    """Train model, tune threshold on validation data, and evaluate on test set."""
    print(f"Training Logistic Regression (class_weight='{class_weight}')...")
    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"  Train Class Distribution: Churn=0: {(y_train == 0).sum():,}, Churn=1: {(y_train == 1).sum():,}")

    model = ChurnLogisticRegression(class_weight=class_weight)
    model.fit(X_train, y_train)

    val_metrics_default = model.evaluate(X_val, y_val, threshold=0.5)

    if tune_threshold:
        best_t, best_score = model.tune_threshold(X_val, y_val, metric=tune_metric)
        print(f"Tuned threshold on validation: {best_t} (Val {tune_metric.upper()}: {best_score})")

    train_metrics = model.evaluate(X_train, y_train)
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test, y_test)

    results = {
        "train": train_metrics,
        "val_default_0.5": val_metrics_default,
        "val_tuned": val_metrics,
        "test": test_metrics,
    }

    print("\n" + format_metrics_summary(test_metrics, title="TEST SET EVALUATION RESULTS"))

    top_features = model.get_feature_importance().head(10)
    print("\nTop 10 Influential Features (by Odds Ratio):")
    print(top_features[["feature", "coefficient", "odds_ratio"]].to_string(index=False))

    if save_path:
        saved_file = model.save(save_path)
        print(f"\nModel successfully saved to: {saved_file}")

    if log_run:
        try:
            run_id = log_experiment(
                model_name="LogisticRegression",
                strategy=f"class_weight='{class_weight}' (tuned)",
                metrics=test_metrics,
                features_count=len(model.feature_names_),
                threshold=test_metrics.get("threshold", 0.5),
                artifacts_path=str(save_path) if save_path else None,
            )
            print(f"Logged run #{run_id} to outputs/reports/experiment_log.csv")
        except Exception as exc:
            print(f"Could not log experiment: {exc}")

    return model, results


def main():
    """Execute model training and evaluation pipeline."""
    print("Loading data from database...")
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_model_ready_data()
    except Exception as e:
        print(f"Could not load data from database: {e}")
        return

    output_file = PROJECT_ROOT / "outputs" / "models" / "churn_logistic_regression.joblib"
    train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        class_weight="balanced",
        tune_threshold=True,
        tune_metric="balanced_accuracy",
        save_path=output_file,
    )


if __name__ == "__main__":
    main()
