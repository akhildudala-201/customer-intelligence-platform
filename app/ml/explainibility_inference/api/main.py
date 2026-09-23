from __future__ import annotations

import os

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.ml.explainibility_inference.api.dependencies import get_predictor
from app.ml.explainibility_inference.api.schemas import (
    BatchPredictRequest,
    ChurnPredictionResponse,
    FeaturesPredictRequest,
)
from app.ml.explainibility_inference.data_access.customer_feature_repository import (
    CustomerNotFoundError,
    get_customer_features,
    get_customer_features_batch,
)
from app.ml.explainibility_inference.inference.predict import ChurnPredictor

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


@app.post(
    f"{API_PREFIX}/predictions/from-features",
    response_model=list[ChurnPredictionResponse],
)
def predict_from_features(
    payload: FeaturesPredictRequest,
    predictor: ChurnPredictor = Depends(get_predictor),
):
    rows = [
        {"customer_unique_id": customer.customer_unique_id, **customer.features}
        for customer in payload.customers
    ]
    return predictor.predict(pd.DataFrame(rows))


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
