"""FastAPI Application for Customer Intelligence Platform - Unified API."""

from __future__ import annotations

import importlib
import logging
import os

from dotenv import load_dotenv

load_dotenv()  # must run before reading API_PREFIX / ALLOWED_ORIGINS below

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("customersphere.api")

API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

# Modular router packages across the customer intelligence platform
ROUTER_MODULES = [
    # Churn dashboard, stored explanations & direct customer scoring
    ("app.api.churn_analytics.router", API_PREFIX),
    # ML Evaluation, Calibration, Experiments & Feature Distributions
    ("app.api.model_evaluation.router", API_PREFIX),
    # Campaigns, Churn Correlations & Forecasting
    ("app.api.campaigns_forecast.router", API_PREFIX),
    # Cohort Analysis & Historical Trends
    ("app.api.cohort_trend_api.routers", API_PREFIX),
    # CLV, Delivery & Payments Drivers
    ("app.api.clv_delivery_payments.clv_delivery_endpoints", API_PREFIX),
    # Segmentation & Risk
    ("app.api.segmentation_and_risk.segmentation_routes", API_PREFIX),
    # Other teammates (optional / future additions)
    ("app.api.ml_predictions.router", API_PREFIX),
    ("app.api.segmentation.router", API_PREFIX),
]

app = FastAPI(
    title=os.getenv("APP_NAME", "Customer Intelligence Platform API"),
    description="Unified API for Churn Model Evaluation, Experiments, Features, Campaigns, Forecasting, Cohorts, and CLV.",
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
    return {"status": "ok", "service": "customer-intelligence-api"}


def _include_routers() -> None:
    for module_path, prefix in ROUTER_MODULES:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, "router"):
                if prefix:
                    app.include_router(module.router, prefix=prefix)
                else:
                    app.include_router(module.router)
                logger.info("Successfully mounted %s under prefix '%s'", module_path, prefix)
        except ModuleNotFoundError as exc:
            # Skip if the router package hasn't been created yet
            if exc.name and module_path.startswith(exc.name):
                logger.warning("Skipping router %s (not created yet)", module_path)
                continue
            logger.exception("Failed to import router from %s: %s", module_path, exc)
        except Exception as exc:
            logger.exception("Error mounting router %s: %s", module_path, exc)


_include_routers()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=port, reload=True)
