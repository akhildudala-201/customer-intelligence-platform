"""
app/ml/inference/__init__.py

Re-exports the inference package's public interface so it can be imported
either at the package level (from app.ml.explainibility_interface.inference import X) or from the
specific submodule (from app.ml.explainibility_interface.inference.predict import X, from
app.ml.inference.feature_contract import Y, etc.) — both styles are used
across this codebase's tests, so this file makes the package-level form
work without needing to touch any test file or the submodules themselves.
"""

from app.ml.explainibility_interface.inference.feature_contract import (
    FEATURE_DTYPES,
    ID_COLUMN,
    REQUIRED_FEATURES,
    FeatureValidationError,
    check_contract_matches_model,
    get_model_feature_names,
    validate_features,
)
from app.ml.explainibility_interface.inference.model_adapter import ChurnModelAdapter
from app.ml.explainibility_interface.inference.model_loader import ModelArtifactNotFoundError, load_model
from app.ml.explainibility_interface.inference.predict import (
    DEFAULT_MODEL_PATH,
    DEFAULT_REASON_CODES_PATH,
    MODEL_VERSION,
    TOP_N_REASON_CODES,
    ChurnPredictor,
    predict,
)

__all__ = [
    "FEATURE_DTYPES",
    "ID_COLUMN",
    "REQUIRED_FEATURES",
    "FeatureValidationError",
    "check_contract_matches_model",
    "get_model_feature_names",
    "validate_features",
    "ChurnModelAdapter",
    "ModelArtifactNotFoundError",
    "load_model",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_REASON_CODES_PATH",
    "MODEL_VERSION",
    "TOP_N_REASON_CODES",
    "ChurnPredictor",
    "predict",
]
