from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


class ChurnShapExplainer:
   
    def __init__(self, shap_model: Any, feature_names: list[str]):
        self.feature_names = feature_names
        self._explainer = shap.TreeExplainer(shap_model)

    def explain(self, X: pd.DataFrame) -> list[dict[str, float]]:
       
        if list(X.columns) != self.feature_names:
            raise ValueError(
                "Column order passed to ChurnShapExplainer.explain() does "
                f"not match expected feature order. Expected "
                f"{self.feature_names}, got {list(X.columns)}."
            )

        raw_shap_values = self._explainer.shap_values(X)

        values = self._normalize_to_positive_class(raw_shap_values)

        return [self._row_to_json(row) for row in values]

    def _normalize_to_positive_class(self, raw_shap_values: Any) -> np.ndarray:
        """Handle the several shapes shap_values() can return and reduce to
        a single (n_samples, n_features) array for the positive class."""
        if isinstance(raw_shap_values, list):
            # [class_0_array, class_1_array] -> take positive class.
            return np.asarray(raw_shap_values[1])

        arr = np.asarray(raw_shap_values)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes) -> take last class as positive.
            return arr[:, :, -1]
        return arr

    def _row_to_json(self, row: np.ndarray) -> dict[str, float]:
        """Convert one row of SHAP values (numpy floats) into a plain
        Python dict with native float values, ready for JSON serialization."""
        return {
            feature: float(value)
            for feature, value in zip(self.feature_names, row)
        }
