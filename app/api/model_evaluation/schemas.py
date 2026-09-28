"""Schemas for model evaluation, calibration, experiments, and feature analysis endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /model/performance-summary
# ---------------------------------------------------------------------------

class ConfusionMatrixSchema(BaseModel):
    tn: int = Field(..., description="True Negatives")
    fp: int = Field(..., description="False Positives")
    fn: int = Field(..., description="False Negatives")
    tp: int = Field(..., description="True Positives")


class PerformanceMetricsSchema(BaseModel):
    accuracy: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    f2: Optional[float] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    log_loss: Optional[float] = None
    lift: Optional[float] = None
    confusion_matrix: Optional[ConfusionMatrixSchema] = None
    threshold: Optional[float] = None


class PerformanceSummaryResponse(BaseModel):
    model_name: str
    version_or_timestamp: Optional[str] = None
    split_evaluated: str = "test"
    metrics: PerformanceMetricsSchema
    operating_point: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# /model/comparison
# ---------------------------------------------------------------------------

class ModelMetricComparisonRow(BaseModel):
    metric: str
    logistic_regression: Optional[float] = None
    lightgbm: Optional[float] = None
    better: Optional[str] = None


class ModelComparisonResponse(BaseModel):
    comparison_table: List[ModelMetricComparisonRow]
    summary_winner: Optional[str] = None
    best_thresholds: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# /model/threshold-analysis
# ---------------------------------------------------------------------------

class ThresholdSweepPoint(BaseModel):
    threshold: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float


class BestThresholdItem(BaseModel):
    metric: str
    best_threshold_logreg: Optional[float] = None
    best_score_logreg: Optional[float] = None
    best_threshold_lgbm: Optional[float] = None
    best_score_lgbm: Optional[float] = None
    better: Optional[str] = None


class ThresholdAnalysisResponse(BaseModel):
    recommended_model: str = "LightGBM"
    sweep_curve: List[ThresholdSweepPoint]
    best_thresholds: List[BestThresholdItem]


# ---------------------------------------------------------------------------
# /model/version
# ---------------------------------------------------------------------------

class ModelVersionResponse(BaseModel):
    model_name: str
    timestamp: str
    features: List[str]
    features_count: int
    operating_point: Dict[str, Any]
    positive_class: int
    best_hyperparameters: Optional[Dict[str, Any]] = None
    artifact_path: Optional[str] = None


# ---------------------------------------------------------------------------
# /model/experiments
# ---------------------------------------------------------------------------

class ExperimentRunRecord(BaseModel):
    run_id: int
    timestamp: str
    model: str
    strategy: str
    features_count: int
    threshold: str
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None


class ExperimentsHistoryResponse(BaseModel):
    total_runs: int
    runs: List[ExperimentRunRecord]


# ---------------------------------------------------------------------------
# /model/calibration
# ---------------------------------------------------------------------------

class CalibrationMethodMetrics(BaseModel):
    method: str
    brier_score: float
    log_loss: float
    ece: float
    mce: float


class CalibrationResponse(BaseModel):
    model_name: str = "LightGBM"
    selected_method: str
    calibration_comparison: List[CalibrationMethodMetrics]
    test_calibration_before: Optional[Dict[str, float]] = None
    test_calibration_after: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# /model/imbalance-experiments
# ---------------------------------------------------------------------------

class ImbalanceStrategyRecord(BaseModel):
    strategy: str
    tuned_threshold: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    retained_recall: Optional[str] = None
    retained_precision: Optional[str] = None
    churn_recall: Optional[str] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    tn: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None
    tp: Optional[int] = None


class ImbalanceExperimentsResponse(BaseModel):
    model: str
    strategies: List[ImbalanceStrategyRecord]


# ---------------------------------------------------------------------------
# /churn/definition
# ---------------------------------------------------------------------------

class ChurnCounts(BaseModel):
    retained: int
    churned: int
    censored: int
    total_customers: int
    churn_rate_uncensored_pct: float
    churn_rate_total_pct: float


class ChurnDefinitionResponse(BaseModel):
    reference_date: str
    return_window_days: int
    exclude_censored_customers: bool
    definition_rule: str
    counts: ChurnCounts


# ---------------------------------------------------------------------------
# /features/summary
# ---------------------------------------------------------------------------

class FeatureStats(BaseModel):
    feature: str
    count: int
    null_count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    p25: Optional[float] = None
    median: Optional[float] = None
    p75: Optional[float] = None
    max: Optional[float] = None
    iqr: Optional[float] = None


class FeaturesSummaryResponse(BaseModel):
    total_records: int
    features: List[FeatureStats]


# ---------------------------------------------------------------------------
# /features/distribution?feature=
# ---------------------------------------------------------------------------

class HistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int
    density: float


class FeatureDistributionResponse(BaseModel):
    feature: str
    total_count: int
    null_count: int
    bins: List[HistogramBin]


# ---------------------------------------------------------------------------
# /features/churn-by-feature?feature=
# ---------------------------------------------------------------------------

class FeatureChurnBucket(BaseModel):
    bucket_label: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    customer_count: int
    churned_count: int
    churn_rate_pct: float


class ChurnByFeatureResponse(BaseModel):
    feature: str
    total_customers: int
    buckets: List[FeatureChurnBucket]
