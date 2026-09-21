from __future__ import annotations

from pydantic import BaseModel, Field


class ChurnPredictionResponse(BaseModel):
    customer_unique_id: str
    churn_probability: float
    shap_values: dict[str, float]
    reason_codes: list[str]
    model_version: str
    scored_at: str


class BatchPredictRequest(BaseModel):
    customer_unique_ids: list[str] = Field(..., min_length=1)
