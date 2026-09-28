from __future__ import annotations

import importlib
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.model_evaluation.router import router as model_evaluation_router

logger = logging.getLogger("customersphere.api")

API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]

ROUTER_MODULES = [
    "app.api.ml_predictions.router",
    "app.api.ml_evaluation.router",
    "app.api.segmentation.router",
    "app.api.clv_drivers.router",
    "app.api.campaigns_forecast.router",
    "app.api.trends_cohorts.router",
]

app = FastAPI(
    title="CustomerSphere Dashboard API",
    description="Customer Intelligence Platform API",
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
    return {"status": "ok"}


def _include_routers() -> None:
    for module_path in ROUTER_MODULES:
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            if exc.name and module_path.startswith(exc.name):
                logger.warning("Skipping %s (not created yet)", module_path)
                continue
            raise

        app.include_router(module.router, prefix=API_PREFIX)
        logger.info("Included %s under %s", module_path, API_PREFIX)


_include_routers()

app.include_router(model_evaluation_router)
