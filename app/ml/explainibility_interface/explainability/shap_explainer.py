"""
shap_explainer.py

WHY THIS FILE EXISTS
---------------------
Isolates all SHAP-specific logic (building the explainer, computing values,
converting numpy/shap types to plain Python types) away from prediction
logic and away from business/reason-code logic. This file knows NOTHING
about reason codes or business text — it only produces numeric feature
contributions. Reason-code mapping happens later, in predict.py /
reason_codes lookup.

WHAT IT ACCEPTS
---------------
`ChurnShapExplainer(shap_model, feature_names)`:
    - shap_model: the raw tree model object (from
      ChurnModelAdapter.get_shap_model()).
    - feature_names: ordered list of feature column names (from
      feature_contract.REQUIRED_FEATURES).

`explain(X)` accepts a pandas DataFrame of features (one or many rows),
containing exactly `feature_names` columns in that order.

WHAT IT RETURNS
----------------
`explain(X)` returns a list of dicts (one per row of X), each mapping
feature name -> plain Python float SHAP value, e.g.:

    [{"recency_days": 0.31, "frequency": -0.08, "monetary_value": 0.05}, ...]

This works uniformly for a single customer (X with 1 row) or a batch.

IF THE MODEL TYPE EVER CHANGES
-------------------------------
Uses shap.TreeExplainer, which is correct for LightGBM (the actual model
type this pipeline trains — see app/ml/train_lightgbm_model.py) and any
other tree-based model (XGBoost, sklearn RandomForest/GBM, etc.) — only
the model object passed in changes (via ChurnModelAdapter.get_shap_model()).
If the deployed model is ever swapped for something NOT tree-based (e.g.
a linear model), swap shap.TreeExplainer for shap.LinearExplainer or
shap.Explainer(model) here; the `explain()` output contract (list of
dicts) stays the same, so predict.py does not need to change.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


class ChurnShapExplainer:
    """
    Wraps a shap.TreeExplainer over the churn model and returns
    JSON-serializable per-row feature contribution dictionaries.
    """

    def __init__(self, shap_model: Any, feature_names: list[str]):
        self.feature_names = feature_names
        self._explainer = shap.TreeExplainer(shap_model)

    def explain(self, X: pd.DataFrame) -> list[dict[str, float]]:
        """
        Compute SHAP values for every row in X.

        Supports both a single customer (X with 1 row) and a batch of many
        customers — the same code path handles both, since shap.TreeExplainer
        naturally vectorizes over rows.

        Returns a list with one dict per row of X, in the same row order,
        mapping feature_name -> SHAP value (plain Python float).
        """
        if list(X.columns) != self.feature_names:
            raise ValueError(
                "Column order passed to ChurnShapExplainer.explain() does "
                f"not match expected feature order. Expected "
                f"{self.feature_names}, got {list(X.columns)}."
            )

        raw_shap_values = self._explainer.shap_values(X)

        # For binary classification, some SHAP/model combinations return a
        # list of two arrays ([class_0_values, class_1_values]); others
        # (newer shap + LightGBM sklearn API) return a single array already
        # corresponding to the positive class, or a 3-D array
        # (n_samples, n_features, n_classes). Normalize all of these to a
        # single 2-D (n_samples, n_features) array for the positive class.
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
