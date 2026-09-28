"""Tests for Model Evaluation, Experimentation, Calibration, and Feature Analytics API endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_model_performance_summary():
    response = client.get("/model/performance-summary")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "LightGBM"
    assert "metrics" in data
    assert "roc_auc" in data["metrics"]
    assert "pr_auc" in data["metrics"]
    assert data["metrics"]["roc_auc"] is not None


def test_model_comparison():
    response = client.get("/model/comparison")
    assert response.status_code == 200
    data = response.json()
    assert "comparison_table" in data
    assert len(data["comparison_table"]) > 0
    first_row = data["comparison_table"][0]
    assert "metric" in first_row
    assert "logistic_regression" in first_row
    assert "lightgbm" in first_row
    assert data["summary_winner"] in ["LightGBM", "Logistic Regression"]


def test_model_threshold_analysis():
    response = client.get("/model/threshold-analysis")
    assert response.status_code == 200
    data = response.json()
    assert data["recommended_model"] == "LightGBM"
    assert "sweep_curve" in data
    assert len(data["sweep_curve"]) > 0
    assert "best_thresholds" in data
    assert len(data["best_thresholds"]) > 0
    assert "threshold" in data["sweep_curve"][0]


def test_model_version():
    response = client.get("/model/version")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "features" in data
    assert len(data["features"]) > 0
    assert "operating_point" in data
    assert "positive_class" in data


def test_model_experiments():
    response = client.get("/model/experiments")
    assert response.status_code == 200
    data = response.json()
    assert "total_runs" in data
    assert "runs" in data
    assert data["total_runs"] > 0
    first_run = data["runs"][0]
    assert "run_id" in first_run
    assert "model" in first_run
    assert "strategy" in first_run

    # Test limit parameter
    resp_limit = client.get("/model/experiments?limit=2")
    assert resp_limit.status_code == 200
    assert len(resp_limit.json()["runs"]) <= 2


def test_model_calibration():
    response = client.get("/model/calibration")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "selected_method" in data
    assert "calibration_comparison" in data
    assert len(data["calibration_comparison"]) > 0
    first_method = data["calibration_comparison"][0]
    assert "method" in first_method
    assert "brier_score" in first_method
    assert "ece" in first_method


def test_model_imbalance_experiments():
    response = client.get("/model/imbalance-experiments?model=lightgbm")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "LightGBM"
    assert "strategies" in data
    assert len(data["strategies"]) > 0
    first_strat = data["strategies"][0]
    assert "strategy" in first_strat
    assert "balanced_accuracy" in first_strat
    assert "roc_auc" in first_strat

    # Test logistic regression query
    resp_lr = client.get("/model/imbalance-experiments?model=logistic_regression")
    assert resp_lr.status_code == 200
    assert resp_lr.json()["model"] == "Logistic Regression"


def test_churn_definition():
    response = client.get("/churn/definition")
    assert response.status_code == 200
    data = response.json()
    assert data["return_window_days"] == 180
    assert "2018-10-17" in data["reference_date"]
    assert "counts" in data
    counts = data["counts"]
    assert counts["retained"] > 0
    assert counts["churned"] > 0
    assert counts["total_customers"] > 0
    assert counts["churn_rate_uncensored_pct"] > 0


def test_features_summary():
    response = client.get("/features/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_records" in data
    assert "features" in data
    assert len(data["features"]) > 0
    stat = data["features"][0]
    assert "mean" in stat
    assert "median" in stat
    assert "iqr" in stat
    assert "min" in stat
    assert "max" in stat


def test_features_distribution():
    response = client.get("/features/distribution?feature=monetary_value&bins=10")
    assert response.status_code == 200
    data = response.json()
    assert data["feature"] == "monetary_value"
    assert "bins" in data
    assert len(data["bins"]) == 10
    first_bin = data["bins"][0]
    assert "bin_start" in first_bin
    assert "bin_end" in first_bin
    assert "count" in first_bin
    assert "density" in first_bin


def test_features_churn_by_feature():
    response = client.get("/features/churn-by-feature?feature=monetary_value&buckets=5")
    assert response.status_code == 200
    data = response.json()
    assert data["feature"] == "monetary_value"
    assert "buckets" in data
    assert len(data["buckets"]) > 0
    first_bucket = data["buckets"][0]
    assert "bucket_label" in first_bucket
    assert "customer_count" in first_bucket
    assert "churn_rate_pct" in first_bucket


def test_features_summary_subset():
    response = client.get("/features/summary?features=monetary_value&features=avg_review_score")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert len(data["features"]) == 2
    returned_names = {f["feature"] for f in data["features"]}
    assert returned_names == {"monetary_value", "avg_review_score"}


def test_features_churn_by_feature_discrete():
    # Test discrete/binary column branch
    response = client.get("/features/churn-by-feature?feature=has_bad_review&buckets=5")
    assert response.status_code == 200
    data = response.json()
    assert data["feature"] == "has_bad_review"
    assert "buckets" in data
    assert len(data["buckets"]) <= 3  # discrete values 0 and 1
    for b in data["buckets"]:
        assert b["churn_rate_pct"] >= 0.0
        assert b["customer_count"] > 0


def test_features_distribution_validation_boundaries():
    # Missing required 'feature' param -> 422
    resp_missing = client.get("/features/distribution")
    assert resp_missing.status_code == 422

    # bins < 2 -> 422
    resp_too_low = client.get("/features/distribution?feature=monetary_value&bins=1")
    assert resp_too_low.status_code == 422

    # bins > 100 -> 422
    resp_too_high = client.get("/features/distribution?feature=monetary_value&bins=101")
    assert resp_too_high.status_code == 422


def test_features_churn_by_feature_validation_boundaries():
    # Missing required 'feature' param -> 422
    resp_missing = client.get("/features/churn-by-feature")
    assert resp_missing.status_code == 422

    # buckets < 2 -> 422
    resp_too_low = client.get("/features/churn-by-feature?feature=monetary_value&buckets=1")
    assert resp_too_low.status_code == 422

    # buckets > 20 -> 422
    resp_too_high = client.get("/features/churn-by-feature?feature=monetary_value&buckets=25")
    assert resp_too_high.status_code == 422


def test_feature_not_found_errors():
    resp_dist = client.get("/features/distribution?feature=invalid_feature_xyz")
    assert resp_dist.status_code in [400, 500]
    assert "Could not calculate distribution" in resp_dist.json()["detail"]

    resp_churn = client.get("/features/churn-by-feature?feature=invalid_feature_xyz")
    assert resp_churn.status_code in [400, 500]
    assert "Could not calculate churn by feature" in resp_churn.json()["detail"]

