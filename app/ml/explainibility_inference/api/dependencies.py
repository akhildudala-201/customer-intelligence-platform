"""
dependencies.py

WHY THIS FILE EXISTS
---------------------
Loading a ChurnPredictor means loading the model artifact, building the
SHAP explainer, and parsing the reason codes YAML — expensive-ish and
meant to happen once, not on every request. This module holds that single
shared instance and hands it out via FastAPI's dependency injection, and
is the one place a test needs to override to swap in a fake predictor.

WHAT IT ACCEPTS
---------------
Nothing — `get_predictor()` takes no arguments.

WHAT IT RETURNS
----------------
The process-wide `ChurnPredictor` instance, built with default (env-aware)
paths on first use and reused after that.

WHICH PART IS DUMMY / REPLACEABLE
-----------------------------------
None of this file — it just wraps whatever ChurnPredictor() resolves to,
based on the MODEL_PATH/MODEL_VERSION env vars described in
app/ml/inference/predict.py.

WHAT CHANGES WHEN PERSON 5 SUPPLIES THE FINAL MODEL
-----------------------------------------------------
Nothing here. Restarting the API process after MODEL_PATH is pointed at
the real artifact is enough to pick it up.
"""

from __future__ import annotations

from app.ml.explainibility_inference.inference.predict import ChurnPredictor

_predictor: ChurnPredictor | None = None


def get_predictor() -> ChurnPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor()
    return _predictor
