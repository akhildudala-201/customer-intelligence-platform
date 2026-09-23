from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd


class SupportsPredictProba(Protocol):

    def predict_proba(self, X: Any) -> np.ndarray: ...


class _CalibratedArtifactShim:
 
    def __init__(self, artifact: dict):
        self._base_model = artifact["base_model"]
        self._calibrator = artifact.get("calibrator")

        self.raw_tree_model = getattr(self._base_model, "sk_model", self._base_model)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.asarray(self._base_model.predict_proba(X)).astype(float)
        if self._calibrator is not None:
            p = np.asarray(self._calibrator.predict_proba(p)).astype(float)
 
        return np.column_stack([1.0 - p, p])


class ChurnModelAdapter:

    _CHURN_ORIGINAL_LABEL = 1


    DEFAULT_DECISION_THRESHOLD = 0.5

    def __init__(self, model: SupportsPredictProba | dict):
        self.feature_names: list[str] | None = None
        if isinstance(model, dict) and "model" in model:

            self._artifact: dict | None = model
            self._model = model["model"]
            feature_names = model.get("feature_cols")
            if feature_names:
                self.feature_names = list(feature_names)

            op = model.get("operating_point") or {}
            self.decision_threshold = (
                float(op["value"])
                if op.get("mode") == "score" and "value" in op
                else self.DEFAULT_DECISION_THRESHOLD
            )
        elif isinstance(model, dict) and "base_model" in model:

            self._artifact = None
            self._model = _CalibratedArtifactShim(model)
            feature_names = model.get("feature_cols")
            if feature_names:
                self.feature_names = list(feature_names)

            self.decision_threshold = model.get("threshold", self.DEFAULT_DECISION_THRESHOLD)
        else:
            self._artifact = None
            self._model = model
            self.decision_threshold = self.DEFAULT_DECISION_THRESHOLD
            feature_names = getattr(model, "feature_names_in_", None)
            if feature_names is None:
                feature_names = getattr(model, "feature_name_", None)
            if feature_names:
                self.feature_names = list(feature_names)

    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:

        if self.feature_names:
            missing = [name for name in self.feature_names if name not in X.columns]
            if missing:
                raise ValueError(
                    "Model artifact expects feature column(s) not present in "
                    f"the input: {missing}"
                )
            X = X[self.feature_names]

        proba = np.asarray(self._model.predict_proba(X))
        positive_class = self._artifact.get("positive_class") if self._artifact else None

        if positive_class is not None and positive_class != self._CHURN_ORIGINAL_LABEL:
            return 1.0 - proba[:, 1]

        # Column 1 = probability of the positive class (churn = 1).
        return proba[:, 1]

    def get_shap_model(self) -> Any:

        if isinstance(self._model, _CalibratedArtifactShim):
            return self._model.raw_tree_model
        return self._model
