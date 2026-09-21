"""
main.py

WHY THIS FILE EXISTS
---------------------
Exposes Person 6's ChurnPredictor over HTTP — the final step of the
architecture doc's data flow (... -> churn_predictions -> FastAPI /
segmentation). No FastAPI app existed anywhere in the repo, so this is a
new, minimal one. It does not modify app/Database, app/Features, or
app/config, and it never writes to the database — it only reads from
`customer_features` through app/ml/data_access/customer_feature_repository.py.

WHAT IT ACCEPTS
---------------
- GET  /health
- GET  {API_PREFIX}/predictions/{customer_unique_id}
      -> pulls that customer's row from MySQL, runs the full pipeline.
- POST {API_PREFIX}/predictions/batch
      body: {"customer_unique_ids": ["C001", "C002", ...]}
      -> pulls all rows in one query, runs the pipeline for each.

API_PREFIX and ALLOWED_ORIGINS are read from the repo's existing .env
convention (see .env.example) so this fits the same configuration story
as the rest of the app, defaulting to "/api/v1" and no CORS if unset.

WHAT IT RETURNS
----------------
JSON churn_predictions record(s) — the exact shape
app/ml/inference/predict.build_prediction_record produces.

WHICH PART IS DUMMY / REPLACEABLE
-----------------------------------
Nothing API-specific is dummy, and there is no dummy model anywhere in
this pipeline — the predictor it wraps loads whatever real, calibrated
artifact MODEL_PATH points at (see app/ml/inference/predict.py).

WHAT CHANGES WHEN PERSON 5 SUPPLIES THE FINAL MODEL
-----------------------------------------------------
Nothing here. Set MODEL_PATH / MODEL_VERSION and restart the process (see
app/ml/inference/predict.py) — ChurnPredictor is built once at first use via
app/api/dependencies.py and will pick up the real model automatically.

RUNNING IT
----------
    uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000

(HOST/PORT also come from .env if you'd rather read them at the shell
level; uvicorn doesn't read them automatically, so pass them explicitly
as above or via `--env-file .env` on newer uvicorn versions.)
"""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.ml.explainibility_interface.api.dependencies import get_predictor
from app.ml.explainibility_interface.api.schemas import (
    BatchPredictRequest,
    ChurnPredictionResponse,
)
from app.ml.explainibility_interface.data_access.customer_feature_repository import (
    CustomerNotFoundError,
    get_customer_features,
    get_customer_features_batch,
)
from app.ml.explainibility_interface.inference.predict import ChurnPredictor

API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [origin.strip() for origin in _allowed_origins.split(",") if origin.strip()]

app = FastAPI(title="CustomerSphere Churn Prediction API")

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get(
    f"{API_PREFIX}/predictions/{{customer_unique_id}}",
    response_model=ChurnPredictionResponse,
)
def predict_one(
    customer_unique_id: str,
    predictor: ChurnPredictor = Depends(get_predictor),
):
    try:
        features_df = get_customer_features(customer_unique_id)
    except CustomerNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown customer_unique_id: {customer_unique_id}",
        )
    return predictor.predict(features_df)[0]


@app.post(
    f"{API_PREFIX}/predictions/batch",
    response_model=list[ChurnPredictionResponse],
)
def predict_batch(
    payload: BatchPredictRequest,
    predictor: ChurnPredictor = Depends(get_predictor),
):
    try:
        features_df = get_customer_features_batch(payload.customer_unique_ids)
    except CustomerNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="None of the requested customer_unique_ids were found.",
        )
    return predictor.predict(features_df)