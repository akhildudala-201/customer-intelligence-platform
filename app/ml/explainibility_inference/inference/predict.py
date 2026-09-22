from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from app.ml.explainibility_inference.config.reason_code_lookup import load_reason_codes, select_top_reason_codes
from app.ml.explainibility_inference.explainability.shap_explainer import ChurnShapExplainer
from app.ml.explainibility_inference.inference.feature_contract import (
    ID_COLUMN,
    REQUIRED_FEATURES,
    check_contract_matches_model,
    validate_features,
)
from app.ml.explainibility_inference.inference.model_adapter import ChurnModelAdapter
from app.ml.explainibility_inference.inference.model_loader import load_model


_DOTENV_PATH = find_dotenv(usecwd=True)
load_dotenv(_DOTENV_PATH)
_THIS_DIR = Path(__file__).resolve().parent.parent  # -> app/ml/
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


_MODEL_PATH_ENV = os.getenv("MODEL_PATH")
if _MODEL_PATH_ENV:
    _configured_model_path = Path(_MODEL_PATH_ENV).expanduser()
    if not _configured_model_path.is_absolute():
        _configured_model_path = _PROJECT_ROOT / _configured_model_path
    DEFAULT_MODEL_PATH = _configured_model_path
else:
    DEFAULT_MODEL_PATH = None

DEFAULT_REASON_CODES_PATH = _THIS_DIR / "config" / "reason_codes.yaml"


MODEL_VERSION = os.getenv("MODEL_VERSION")

TOP_N_REASON_CODES = 3


def _current_timestamp() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def build_prediction_record(
    customer_unique_id: str,
    churn_probability: float,
    shap_values: dict[str, float],
    reason_codes: list[str],
    model_version: str,
    scored_at: str | None = None,
) -> dict:
   
    return {
        "customer_unique_id": customer_unique_id,
        "churn_probability": round(float(churn_probability), 6),
        "shap_values": shap_values,
        "reason_codes": reason_codes,
        "model_version": model_version,
        "scored_at": scored_at or _current_timestamp(),
    }


class ChurnPredictor:

    def __init__(
        self,
        model_path: str | Path | None = DEFAULT_MODEL_PATH,
        reason_codes_path: str | Path = DEFAULT_REASON_CODES_PATH,
        model_version: str | None = MODEL_VERSION,
        top_n_reason_codes: int = TOP_N_REASON_CODES,
    ):
        if model_path is None:
            raise RuntimeError(
                "No model_path given and MODEL_PATH is not set in the "
                "environment. Set MODEL_PATH in .env to the trained model "
                "artifact's path — there is no dummy model to fall back to."
            )
        if model_version is None:
            raise RuntimeError(
                "No model_version given and MODEL_VERSION is not set in "
                "the environment. Set MODEL_VERSION in .env to identify "
                "the deployed model."
            )

        raw_model = load_model(model_path)
        check_contract_matches_model(raw_model)
        self._adapter = ChurnModelAdapter(raw_model)
        model_feature_names = self._adapter.feature_names or REQUIRED_FEATURES
        self._explainer = ChurnShapExplainer(
            self._adapter.get_shap_model(), feature_names=model_feature_names
        )
        self._reason_code_config = load_reason_codes(reason_codes_path)
        self._model_version = model_version
        self._top_n_reason_codes = top_n_reason_codes

    def predict(self, customer_features: pd.DataFrame) -> list[dict]:
 
        model_feature_names = self._adapter.feature_names or REQUIRED_FEATURES
        validated = validate_features(
            customer_features,
            required_features=model_feature_names,
        )

        ids = validated[ID_COLUMN].tolist()
        X = validated[model_feature_names]

        probabilities = self._adapter.predict_probability(X)
        shap_records = self._explainer.explain(X)

        results = []
        for customer_id, probability, shap_values in zip(
            ids, probabilities, shap_records
        ):
   
            reason_codes = select_top_reason_codes(
                shap_values,
                self._reason_code_config,
                top_n=self._top_n_reason_codes,
                predicted_positive=(probability >= self._adapter.decision_threshold),
            )
            record = build_prediction_record(
                customer_unique_id=customer_id,
                churn_probability=probability,
                shap_values=shap_values,
                reason_codes=reason_codes,
                model_version=self._model_version,
            )
            results.append(record)

        return results


def predict(customer_features: pd.DataFrame) -> list[dict]:
 
    return ChurnPredictor().predict(customer_features)
