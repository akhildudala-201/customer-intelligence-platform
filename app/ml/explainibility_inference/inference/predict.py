"""
predict.py

WHY THIS FILE EXISTS
---------------------
This is the clean, public inference interface for the rest of the system
(FastAPI, downstream segmentation module, batch jobs, tests). It
orchestrates the full pipeline described in the architecture doc:

    model artifact -> model_loader -> model_adapter -> feature validation
    -> predict_proba -> SHAP explainer -> SHAP JSON -> Top-N SHAP features
    -> reason code mapping -> churn_predictions

`build_prediction_record()` (below) used to live in its own module,
output/prediction_output.py — folded in here since it was a single ~15
line function with exactly one caller (this file) and no dedicated tests
of its own; a whole subpackage (folder + __init__.py + file) for that
wasn't earning its keep. If it ever grows a second caller or its own
non-trivial logic, it can move back out.

WHAT IT ACCEPTS
---------------
`ChurnPredictor(model_path, reason_codes_path, model_version)` — paths to
the model artifact and reason codes YAML, plus the model's version string.
model_path and model_version have no built-in default and MUST come from
either the MODEL_PATH / MODEL_VERSION env vars or explicit constructor
arguments — there is no dummy/placeholder model to silently fall back to.

`predict(customer_features)` accepts a pandas DataFrame with one or more
customer rows, containing the ID column + all REQUIRED_FEATURES columns
(extra columns are tolerated and dropped by feature validation).

WHAT IT RETURNS
----------------
`predict()` returns a list of churn_predictions records (dicts), one per
input row, in the shape built by build_prediction_record() below. Works
identically for a single customer (1-row DataFrame) and a batch.
"""

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


# Load the project-wide .env before reading MODEL_PATH/MODEL_VERSION below —
# nothing else in this import chain does this (app.Database.database does
# its own, separate load, but this module doesn't import that at module
# level). find_dotenv(usecwd=True) walks upward from the current working
# directory so this works regardless of which folder a script/pytest run
# starts from, matching the same pattern used in app/Database/database.py
# and machine_learning_part/database_connection/db_connection.py.
# override=False (load_dotenv's default) means real environment variables
# you've already set always win over .env.
_DOTENV_PATH = find_dotenv(usecwd=True)
load_dotenv(_DOTENV_PATH)
_THIS_DIR = Path(__file__).resolve().parent.parent  # -> app/ml/
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Required: no dummy artifact to fall back to. Set MODEL_PATH in .env to
# the real calibrated model artifact.
_MODEL_PATH_ENV = os.getenv("MODEL_PATH")
if _MODEL_PATH_ENV:
    _configured_model_path = Path(_MODEL_PATH_ENV).expanduser()
    if not _configured_model_path.is_absolute():
        _configured_model_path = _PROJECT_ROOT / _configured_model_path
    DEFAULT_MODEL_PATH = _configured_model_path
else:
    DEFAULT_MODEL_PATH = None

DEFAULT_REASON_CODES_PATH = _THIS_DIR / "config" / "reason_codes.yaml"

# Required: no placeholder version string. Set MODEL_VERSION in .env to
# whatever identifies the currently-deployed trained model.
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
    """
    Build a single churn_predictions output record:

        {
            "customer_unique_id": "C001",
            "churn_probability": 0.82,
            "shap_values": {"recency_days": 0.31, "frequency": -0.08, ...},
            "reason_codes": ["RC01", "RC03"],
            "model_version": "v1.0",
            "scored_at": "2026-09-17T12:00:00+00:00"
        }

    `model_version` is required (no default) — pass whatever identifies
    the deployed model (ChurnPredictor always passes its own
    model_version). A caller that omits it gets a TypeError here, not a
    silently mislabeled record.

    `scored_at` is generated at call time (current UTC timestamp) unless
    explicitly provided, which is useful for deterministic tests.
    """
    return {
        "customer_unique_id": customer_unique_id,
        "churn_probability": round(float(churn_probability), 6),
        "shap_values": shap_values,
        "reason_codes": reason_codes,
        "model_version": model_version,
        "scored_at": scored_at or _current_timestamp(),
    }


class ChurnPredictor:
    """
    Public inference interface. Construct once (loads the model + reason
    code config), then call `.predict(df)` as many times as needed.
    """

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
        """
        Run the full inference pipeline for one or many customers.

        Steps: validate input -> churn probability -> SHAP values ->
        SHAP JSON -> top-N reason codes -> assembled output record(s).
        """
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
            # self._adapter.decision_threshold is the calibrated artifact's
            # own re-tuned probability cutoff (from Person 4's threshold
            # analysis, via calibration.py's find_optimal_threshold()) —
            # not a hardcoded guess. See model_adapter.py's docstring for
            # what it falls back to when the loaded artifact isn't a
            # calibrated one.
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
    """
    Module-level convenience function for simple/one-off use. Internally
    builds a ChurnPredictor using default (env-configured) paths on every
    call. For repeated predictions, prefer constructing a single
    ChurnPredictor and reusing it — more efficient, model loaded once.
    """
    return ChurnPredictor().predict(customer_features)
