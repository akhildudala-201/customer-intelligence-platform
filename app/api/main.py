from fastapi import FastAPI

from app.api.segmentation_and_risk.segmentation_routes import (
    router as segmentation_router,
)


app = FastAPI(
    title="Customer Intelligence Platform API",
    description=(
        "Backend APIs for customer segmentation, "
        "customer risk analysis, customer profiles, "
        "customer search, and value analysis."
    ),
    version="1.0.0",
)


app.include_router(
    segmentation_router
)




@app.get(
    "/",
    tags=["Health Check"],
    summary="Check whether the API is running",
)
def root():
    return {
        "message": "Customer Intelligence Platform API is running",
        "status": "healthy",
        "version": "1.0.0",
    }
