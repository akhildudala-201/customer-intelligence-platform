from __future__ import annotations

from app.ml.explainability_inference.inference.predict import ChurnPredictor

_predictor: ChurnPredictor | None = None


def get_predictor() -> ChurnPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor()
    return _predictor
