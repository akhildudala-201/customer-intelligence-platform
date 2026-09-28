"""FastAPI Application for Customer Intelligence Platform - Model Evaluation & Analytics API."""

from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.model_evaluation.router import router as model_evaluation_router

_allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [origin.strip() for origin in _allowed_origins.split(",") if origin.strip()]

app = FastAPI(
    title="Customer Intelligence Platform - Model Evaluation & Analysis API",
    description="API for Churn Model Performance, Threshold Analysis, Experiments History, Calibration, and Feature Distributions.",
    version="1.0.0",
)

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health", tags=["Health"])
def health() -> dict:
    return {"status": "ok", "service": "model-evaluation-api"}


# Mount modular routers:
app.include_router(model_evaluation_router)
