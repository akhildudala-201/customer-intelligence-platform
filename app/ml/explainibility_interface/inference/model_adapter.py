"""
model_adapter.py

WHY THIS FILE EXISTS
---------------------
This is the ONE place in the codebase that is allowed to know anything
about the concrete model implementation (LightGBM, its API, its training
choices). Everything downstream (predict.py, shap_explainer.py, tests)
talks to a ChurnModelAdapter object instead of a raw model, through two
methods:

    predict_probability(X)  -> churn probability
    get_shap_model()        -> the underlying model object SHAP needs

WHAT IT ACCEPTS
---------------
`ChurnModelAdapter(model)` wraps whatever model_loader.load_model()
returned. Three shapes are understood:
  - A bare fitted model.
  - A dict artifact `{"model": <fitted model>, "feature_cols": [...],
    "operating_point": {...}, "positive_class": ..., ...}` — the shape
    train_lightgbm_model.py's save_model() writes for an UNCALIBRATED
    training artifact. The adapter unwraps `artifact["model"]` and uses
    that as the working model, and applies `positive_class` inversion
    here (see predict_probability()).
  - A dict artifact `{"base_model": <_LGBAdapter>, "calibrator": <None |
    PlattScaling | IsotonicCalibration>, "method": "none"|"sigmoid"|
    "isotonic", "threshold": ..., "model_name": ..., "feature_cols":
    [...]}` — the shape calibration.py's CalibratedChurnModel.save()
    writes for the FINAL, deployable artifact (Person 5's calibration
    pipeline — see app/ml/calibration.py). `base_model` here is Person
    5's `_LGBAdapter`, whose own predict_proba(X) already returns a 1-D
    array oriented to P(churn_label == 1) — `_LGBAdapter` corrects
    `positive_class` itself, once, at calibration time (see its
    docstring in calibration.py), so this shape carries NO
    `positive_class` key and needs no re-inversion here.
    `calibrator`, if not None, is Person 5's PlattScaling or
    IsotonicCalibration instance and is applied on top via
    `_CalibratedArtifactShim`.
  In both dict cases the rest of the pipeline never sees the dict itself.

`predict_probability(X)` accepts a pandas DataFrame of validated features
(no ID column — see predict.py for where the ID column is stripped).

WHAT IT RETURNS
----------------
`predict_probability(X)` returns a 1-D numpy array of churn probabilities,
one per row of X, in [0, 1].

`get_shap_model()` returns the raw underlying model object, for use by
shap.TreeExplainer. For a calibrated artifact this is the base LightGBM
estimator (unwrapped past `_LGBAdapter` and the calibrator), since SHAP
needs the actual tree model, not the calibration wrapper.

`decision_threshold` (public attribute, set in __init__) is the
probability cutoff callers should use to turn a probability into a
predicted class. For a calibrated artifact, this is the REAL re-tuned
threshold calibration.py chose via find_optimal_threshold() — not a
placeholder. For every other shape, it falls back to
DEFAULT_DECISION_THRESHOLD (0.5), since no safely-reusable cutoff can be
recovered from those artifacts at request time (see __init__ for why).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd


class SupportsPredictProba(Protocol):
    """Structural type for anything with a scikit-learn-style predict_proba."""

    def predict_proba(self, X: Any) -> np.ndarray: ...


class _CalibratedArtifactShim:
    """
    Adapts calibration.py's CalibratedChurnModel.save() dict shape
    (`{"base_model": ..., "calibrator": ..., ...}`) to the sklearn-style
    2-D predict_proba(X) -> (n_samples, 2) interface ChurnModelAdapter
    expects from `self._model`, so ChurnModelAdapter.predict_probability()
    doesn't need a separate code path for calibrated vs. uncalibrated
    artifacts.

    `base_model` here is Person 5's `_LGBAdapter`, whose predict_proba(X)
    already returns a 1-D array oriented to P(churn_label == 1)
    (positive_class already corrected at calibration time — see
    _LGBAdapter's docstring in calibration.py). `calibrator`, if present,
    maps that 1-D array through Platt/Isotonic scaling, still 1-D. This
    shim only reshapes that 1-D result into the 2-D form the rest of this
    file already knows how to index (`proba[:, 1]`) — it does not
    re-derive or re-check orientation itself.
    """

    def __init__(self, artifact: dict):
        self._base_model = artifact["base_model"]
        self._calibrator = artifact.get("calibrator")
        # Person 5's _LGBAdapter exposes the real fitted estimator as
        # `.sk_model`; duck-typed rather than imported, to keep this
        # serving-layer file decoupled from app/ml/calibration.py (see
        # app/ml/__init__.py's note on why training and serving don't
        # import each other). If base_model isn't an _LGBAdapter (e.g. a
        # bare estimator was wrapped directly), it IS the raw model.
        self.raw_tree_model = getattr(self._base_model, "sk_model", self._base_model)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.asarray(self._base_model.predict_proba(X)).astype(float)
        if self._calibrator is not None:
            p = np.asarray(self._calibrator.predict_proba(p)).astype(float)
        # Reshape the 1-D P(churn==1) array into sklearn's 2-D
        # (n_samples, 2) convention so ChurnModelAdapter's existing
        # `proba[:, 1]` indexing works unchanged for this shape too.
        return np.column_stack([1.0 - p, p])


class ChurnModelAdapter:
    """
    Wraps the raw churn model and exposes a stable interface to the rest of
    the pipeline. See module docstring for why this exists.
    """

    # The label this whole system means by "churn" — matches
    # build_churn_label.py's convention: label 1 = churned, label 0 =
    # retained. Used to detect when the trained model's internal positive
    # class (see predict_probability()) is the opposite one.
    _CHURN_ORIGINAL_LABEL = 1

    # Used only when a real, safely-interpretable decision threshold
    # can't be recovered from the artifact — see decision_threshold below.
    DEFAULT_DECISION_THRESHOLD = 0.5

    def __init__(self, model: SupportsPredictProba | dict):
        if isinstance(model, dict) and "model" in model:
            # Uncalibrated training artifact (train_lightgbm_model.py /
            # logistic_regression.py's save()). May still need
            # positive_class inversion — see predict_probability().
            self._artifact: dict | None = model
            self._model = model["model"]
            # train_lightgbm_model.py's "operating_point" can be a
            # population RATE ("flag the riskiest TARGET_FLAG_RATE% of
            # customers" — CONFIG["THRESHOLD_MODE"] defaults to "rate"),
            # not a probability cutoff. A rate isn't a 0-1 threshold this
            # adapter can reuse directly at request time without knowing
            # the current scoring population's score distribution, so
            # this shape always falls back to the default rather than
            # guessing. (If "operating_point"["mode"] == "score", its
            # value genuinely is a probability cutoff — used when present.)
            op = model.get("operating_point") or {}
            self.decision_threshold = (
                float(op["value"])
                if op.get("mode") == "score" and "value" in op
                else self.DEFAULT_DECISION_THRESHOLD
            )
        elif isinstance(model, dict) and "base_model" in model:
            # Final calibrated artifact (Person 5's calibration.py ->
            # CalibratedChurnModel.save()). Already correctly oriented
            # upstream by _LGBAdapter — no positive_class key, no
            # re-inversion needed, so self._artifact stays None below.
            self._artifact = None
            self._model = _CalibratedArtifactShim(model)
            # This IS a real, directly-usable probability cutoff — the
            # threshold calibration.py re-tuned via find_optimal_threshold()
            # on the CALIBRATED validation probabilities (see
            # calibrate_model() in calibration.py), against whatever
            # metric Person 4's threshold_analysis.py recommended. Safe to
            # use as-is, unlike the rate-based operating_point above.
            self.decision_threshold = model.get("threshold", self.DEFAULT_DECISION_THRESHOLD)
        else:
            self._artifact = None
            self._model = model
            self.decision_threshold = self.DEFAULT_DECISION_THRESHOLD

    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return the probability of churn (per build_churn_label.py: label 1)
        for each row of X.

        train_lightgbm_model.py's CONFIG["POSITIVE_CLASS"] can be None
        (auto-detects whichever original label is rarer in the training
        data and fits the model with THAT remapped to internal class 1) or
        an explicit label. Either way, the saved artifact records which
        original label was used as `positive_class`. If that's not
        _CHURN_ORIGINAL_LABEL (1) — e.g. auto-detection picked "retained"
        because it happened to be the rarer class in this dataset —
        predict_proba()[:, 1] is actually P(retained), so it's inverted
        here. This keeps the adapter correct regardless of what
        POSITIVE_CLASS resolves to on any given retrain.
        """
        proba = np.asarray(self._model.predict_proba(X))
        positive_class = self._artifact.get("positive_class") if self._artifact else None

        if positive_class is not None and positive_class != self._CHURN_ORIGINAL_LABEL:
            return 1.0 - proba[:, 1]

        # Column 1 = probability of the positive class (churn = 1).
        return proba[:, 1]

    def get_shap_model(self) -> Any:
        """
        Return the underlying model object that shap.TreeExplainer should
        be built from.

        For a calibrated artifact (self._model is a _CalibratedArtifactShim),
        returns the base LightGBM estimator — the calibrator itself isn't
        a tree model and SHAP shouldn't see it. For every other shape,
        returns self._model directly, as before.
        """
        if isinstance(self._model, _CalibratedArtifactShim):
            return self._model.raw_tree_model
        return self._model