"""FastAPI router endpoints for Model Evaluation, Experiments, Churn Definitions, and Feature Analysis."""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from app.api.model_evaluation.schemas import (
    CalibrationResponse,
    ChurnByFeatureResponse,
    ChurnDefinitionResponse,
    ExperimentsHistoryResponse,
    FeatureDistributionResponse,
    FeaturesSummaryResponse,
    ImbalanceExperimentsResponse,
    ModelComparisonResponse,
    ModelVersionResponse,
    PerformanceSummaryResponse,
    ThresholdAnalysisResponse,
)
from app.api.model_evaluation import services

router = APIRouter()


# ---------------------------------------------------------------------------
# 1. /model/performance-summary
# ---------------------------------------------------------------------------
@router.get(
    "/model/performance-summary",
    response_model=PerformanceSummaryResponse,
    summary="Get performance metrics for the active champion model",
    tags=["Model Evaluation"],
)
def get_performance_summary():
    """Retrieve test-set metrics (ROC-AUC, PR-AUC, F1, Precision, Recall, confusion matrix) for champion model."""
    try:
        return services.get_model_performance_summary()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load performance summary: {exc}")


# ---------------------------------------------------------------------------
# 2. /model/comparison
# ---------------------------------------------------------------------------
@router.get(
    "/model/comparison",
    response_model=ModelComparisonResponse,
    summary="Compare champion and benchmark models",
    tags=["Model Evaluation"],
)
def get_comparison():
    """Benchmark table comparing Logistic Regression vs LightGBM across all key metrics."""
    try:
        return services.get_model_comparison()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model comparison: {exc}")


# ---------------------------------------------------------------------------
# 3. /model/threshold-analysis
# ---------------------------------------------------------------------------
@router.get(
    "/model/threshold-analysis",
    response_model=ThresholdAnalysisResponse,
    summary="Analyze decision thresholds and tradeoff curves",
    tags=["Model Evaluation"],
)
def get_threshold_analysis():
    """Precision-recall and F1 trade-off across decision thresholds [0.01 to 0.99] and optimal thresholds."""
    try:
        return services.get_threshold_analysis()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load threshold analysis: {exc}")


# ---------------------------------------------------------------------------
# 4. /model/version
# ---------------------------------------------------------------------------
@router.get(
    "/model/version",
    response_model=ModelVersionResponse,
    summary="Get active model metadata, parameters, and artifact path",
    tags=["Model Evaluation"],
)
def get_version():
    """Active model architecture, hyperparameters, features used, operating point, and artifact location."""
    try:
        return services.get_model_version_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model version metadata: {exc}")


# ---------------------------------------------------------------------------
# 5. /model/experiments
# ---------------------------------------------------------------------------
@router.get(
    "/model/experiments",
    response_model=ExperimentsHistoryResponse,
    summary="Get full ML experiment run history from experiment_log.csv",
    tags=["Experiments"],
)
def get_experiments(
    limit: Optional[int] = Query(None, description="Maximum number of latest runs to return"),
):
    """Retrieve recorded machine learning experiment runs from experiment_log.csv."""
    try:
        return services.get_experiments_history(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load experiments history: {exc}")


# ---------------------------------------------------------------------------
# 6. /model/calibration
# ---------------------------------------------------------------------------
@router.get(
    "/model/calibration",
    response_model=CalibrationResponse,
    summary="Calibration comparison and Expected Calibration Error (ECE)",
    tags=["Model Evaluation"],
)
def get_calibration():
    """Probability calibration diagnostics: Brier score, log loss, ECE, and MCE comparison across methods."""
    try:
        return services.get_calibration_details()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load calibration diagnostics: {exc}")


# ---------------------------------------------------------------------------
# 7. /model/imbalance-experiments
# ---------------------------------------------------------------------------
@router.get(
    "/model/imbalance-experiments",
    response_model=ImbalanceExperimentsResponse,
    summary="Benchmark results across class imbalance handling strategies",
    tags=["Experiments"],
)
def get_imbalance_experiments(
    model: str = Query("lightgbm", description="Model family: 'lightgbm' or 'logistic_regression'"),
):
    """Results of class-weighting, undersampling, and oversampling strategies."""
    try:
        return services.get_imbalance_experiments(model=model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load imbalance experiments: {exc}")


# ---------------------------------------------------------------------------
# 8. /churn/definition
# ---------------------------------------------------------------------------
@router.get(
    "/churn/definition",
    response_model=ChurnDefinitionResponse,
    summary="Churn definition criteria, observation window, and customer counts",
    tags=["Churn Definitions"],
)
def get_churn_definition():
    """How churn is defined (return window, reference date) and retained vs churned vs censored customer counts."""
    try:
        return services.get_churn_definition_and_counts()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load churn definition: {exc}")


# ---------------------------------------------------------------------------
# 9. /features/summary
# ---------------------------------------------------------------------------
@router.get(
    "/features/summary",
    response_model=FeaturesSummaryResponse,
    summary="Descriptive statistics (mean, median, IQR, min, max) of key features",
    tags=["Feature Analysis"],
)
def get_features_summary(
    features: Optional[List[str]] = Query(None, description="Optional list of feature names to summarize"),
):
    """Mean, standard deviation, median, 25th/75th percentiles, IQR, min, max, and missing counts."""
    try:
        return services.get_features_summary(features=features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to compute feature summary: {exc}")


# ---------------------------------------------------------------------------
# 10. /features/distribution
# ---------------------------------------------------------------------------
@router.get(
    "/features/distribution",
    response_model=FeatureDistributionResponse,
    summary="Histogram bins and frequencies for any feature",
    tags=["Feature Analysis"],
)
def get_feature_distribution(
    feature: str = Query(..., description="Name of the feature column"),
    bins: int = Query(10, ge=2, le=100, description="Number of histogram bins"),
):
    """Binned histogram distribution for any requested feature in the dataset."""
    try:
        return services.get_feature_distribution(feature_name=feature, num_bins=bins)
    except Exception as exc:
        raise HTTPException(
            status_code=400 if "Unknown column" in str(exc) else 500,
            detail=f"Could not calculate distribution for feature '{feature}': {exc}",
        )


# ---------------------------------------------------------------------------
# 11. /features/churn-by-feature
# ---------------------------------------------------------------------------
@router.get(
    "/features/churn-by-feature",
    response_model=ChurnByFeatureResponse,
    summary="Churn rate per bucket / category for any feature",
    tags=["Feature Analysis"],
)
def get_churn_by_feature(
    feature: str = Query(..., description="Name of the feature column"),
    buckets: int = Query(5, ge=2, le=20, description="Number of quantile/range buckets"),
):
    """Churn rate percentage across value intervals or categories of a given feature."""
    try:
        return services.get_churn_by_feature(feature_name=feature, num_buckets=buckets)
    except Exception as exc:
        raise HTTPException(
            status_code=400 if "Unknown column" in str(exc) else 500,
            detail=f"Could not calculate churn by feature '{feature}': {exc}",
        )
