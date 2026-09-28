import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.segmentation_and_risk import segmentation_routes


client = TestClient(app)
API_PREFIX = "/api/v1"



@pytest.fixture
def mock_customer_data():
    """
    Sample customer data used for API unit tests.
    No real database is required.
    """

    return pd.DataFrame(
        [
            {
                "customer_unique_id": "C001",
                "churn_probability": 0.95,
                "segment_id": 0,
                "segment_label": "High-Value Satisfied Repeat Buyers",
                "risk_tier": "High Risk",
                "cluster_probability": 0.90,
                "clv": 200.0,
                "value_tier": "High",
                "monetary_value": 500.0,
                "frequency": 5,
                "recency_days": 20,
            },
            {
                "customer_unique_id": "C002",
                "churn_probability": 0.60,
                "segment_id": 1,
                "segment_label": "Satisfied One-Time Buyers",
                "risk_tier": "Medium Risk",
                "cluster_probability": 0.80,
                "clv": 100.0,
                "value_tier": "Medium",
                "monetary_value": 200.0,
                "frequency": 2,
                "recency_days": 100,
            },
            {
                "customer_unique_id": "C003",
                "churn_probability": 0.20,
                "segment_id": 1,
                "segment_label": "Satisfied One-Time Buyers",
                "risk_tier": "Low Risk",
                "cluster_probability": 0.70,
                "clv": 50.0,
                "value_tier": "Low",
                "monetary_value": 80.0,
                "frequency": 1,
                "recency_days": 30,
            },
            {
                "customer_unique_id": "C004",
                "churn_probability": 0.80,
                "segment_id": 0,
                "segment_label": "High-Value Satisfied Repeat Buyers",
                "risk_tier": "High Risk",
                "cluster_probability": 0.95,
                "clv": 150.0,
                "value_tier": "High",
                "monetary_value": 400.0,
                "frequency": 4,
                "recency_days": 60,
            },
        ]
    )



@pytest.fixture
def mock_customer_view(monkeypatch, mock_customer_data):
    """
    Replace the real database-backed customer view
    with test data.
    """

    monkeypatch.setattr(
        segmentation_routes,
        "build_customer_view",
        lambda: mock_customer_data.copy(),
    )



def test_get_customers_by_segment(mock_customer_view):

    response = client.get(
        f"{API_PREFIX}/customers/bysegment",
        params={
            "segment_id": 0,
            "limit": 10,
            "offset": 0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 2
    assert data["limit"] == 10
    assert data["offset"] == 0

    assert len(data["customers"]) == 2

    for customer in data["customers"]:
        assert customer["segment_id"] == 0



def test_get_segment_profiles(mock_customer_view):

    response = client.get(
        f"{API_PREFIX}/segments/profile"
    )

    assert response.status_code == 200

    data = response.json()

    assert "segments" in data
    assert len(data["segments"]) == 2

    segment_ids = {
        segment["segment_id"]
        for segment in data["segments"]
    }

    assert segment_ids == {0, 1}



def test_get_segment_customers(mock_customer_view):

    response = client.get(
        f"{API_PREFIX}/segments/0/customers",
        params={
            "limit": 10,
            "offset": 0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["segment_id"] == 0
    assert data["total"] == 2
    assert len(data["customers"]) == 2

    for customer in data["customers"]:
        assert customer["segment_id"] == 0



def test_get_segment_risk_mix(mock_customer_view):

    response = client.get(
        f"{API_PREFIX}/segments/risk-mix"
    )

    assert response.status_code == 200

    data = response.json()

    assert "risk_mix" in data
    assert len(data["risk_mix"]) > 0

    for item in data["risk_mix"]:
        assert "segment_id" in item
        assert "risk_tier" in item
        assert "customer_count" in item




def test_get_risk_summary(mock_customer_view):

    response = client.get(
        f"{API_PREFIX}/risk/summary"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_customers"] == 4

    assert (
        "average_churn_probability"
        in data
    )

    assert (
        "risk_distribution"
        in data
    )

    assert len(
        data["risk_distribution"]
    ) > 0



def test_get_at_risk_customers(mock_customer_view):

    response = client.get(
        f"{API_PREFIX}/atrisk",
        params={
            "risk_tier": "High Risk",
            "min_probability": 0.7,
            "limit": 10,
            "offset": 0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 2

    assert len(
        data["customers"]
    ) == 2

    for customer in data["customers"]:

        assert customer["risk_tier"] == "High Risk"

        assert (
            customer["churn_probability"]
            >= 0.7
        )



def test_get_customer_profile(mock_customer_view):

    response = client.get(
        f"{API_PREFIX}/customers/C001/profile"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["customer_id"] == "C001"

    assert "profile" in data

    assert (
        data["profile"]["customer_unique_id"]
        == "C001"
    )

    assert (
        data["profile"]["risk_tier"]
        == "High Risk"
    )



def test_get_customer_profile_not_found(
    mock_customer_view,
):

    response = client.get(
        f"{API_PREFIX}/customers/UNKNOWN/profile"
    )

    assert response.status_code == 404

    data = response.json()

    assert (
        "not found"
        in data["detail"].lower()
    )



def test_get_segment_summary(mock_customer_view):

    response = client.get(
        f"{API_PREFIX}/segments/summary"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_customers"] == 4

    assert data["segment_count"] == 2

    assert "segments" in data

    assert len(data["segments"]) == 2

    total_segment_customers = sum(
        segment["customer_count"]
        for segment in data["segments"]
    )

    assert (
        total_segment_customers
        == 4
    )



def test_get_segment_value_mix(mock_customer_view):

    response = client.get(
        f"{API_PREFIX}/segments/value-mix"
    )

    assert response.status_code == 200

    data = response.json()

    assert "value_mix" in data

    assert len(
        data["value_mix"]
    ) > 0

    for item in data["value_mix"]:

        assert "segment_id" in item

        assert "value_tier" in item

        assert "customer_count" in item



def test_get_risk_thresholds():

    response = client.get(
        f"{API_PREFIX}/risk/thresholds"
    )

    assert response.status_code == 200

    data = response.json()

    assert "thresholds" in data

    assert "source" in data

    assert len(
        data["thresholds"]
    ) > 0

    for threshold in data["thresholds"]:

        assert "name" in threshold

        assert "upper_bound" in threshold

        assert isinstance(
            threshold["upper_bound"],
            (int, float),
        )



def test_search_customers(mock_customer_view):

    response = client.get(
        f"{API_PREFIX}/customers/search",
        params={
            "q": "C001",
            "limit": 10,
            "offset": 0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "C001"

    assert data["total"] == 1

    assert len(
        data["customers"]
    ) == 1

    assert (
        data["customers"][0][
            "customer_unique_id"
        ]
        == "C001"
    )


# ============================================================
# 11B. CUSTOMER SEARCH WITH RISK FILTER
# ============================================================

def test_search_customers_with_risk_filter(
    mock_customer_view,
):

    response = client.get(
        f"{API_PREFIX}/customers/search",
        params={
            "q": "C",
            "risk_tier": "High Risk",
            "limit": 10,
            "offset": 0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 2

    for customer in data["customers"]:

        assert (
            customer["risk_tier"]
            == "High Risk"
        )


# ============================================================
# PAGINATION TEST
# ============================================================

def test_customer_search_pagination(
    mock_customer_view,
):

    response = client.get(
        f"{API_PREFIX}/customers/search",
        params={
            "q": "C",
            "limit": 2,
            "offset": 0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 4

    assert data["limit"] == 2

    assert data["offset"] == 0

    assert len(
        data["customers"]
    ) == 2
