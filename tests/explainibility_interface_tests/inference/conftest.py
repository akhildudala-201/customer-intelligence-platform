"""
conftest.py (app/ml/tests/explainibility_inference_tests/inference/)

Shared fixtures for the explainability/integration (Person 6) test suite.
Kept in its own app/ml/tests/explainibility_inference_tests/inference/
package, separate from the top-level tests/ (owned by Feature
Engineering), so this module can be tested independently without
touching other interns' test setup. Sys.path setup lives in
app/ml/tests/explainibility_inference_tests/conftest.py, a parent of
this file, and applies here automatically via pytest's normal
conftest.py inheritance — this file holds only fixtures specific to
inference tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.ml.explainibility_inference.inference.predict import ChurnPredictor, DEFAULT_MODEL_PATH, DEFAULT_REASON_CODES_PATH

# Real customer_unique_id format from the Olist dataset (32-char hex, see
# data/olist_customers_dataset.csv) rather than a placeholder ID, so tests
# read like they operate on real pipeline data.
_SAMPLE_ID_1 = "861eff4711a542e4b93843c6dd7febb0"
_SAMPLE_ID_2 = "290c77bc529b7ac935b93aa66c333dc3"
_SAMPLE_ID_3 = "4e7b3e00288586ebd08712fdd0374a03"


@pytest.fixture
def sample_customer_df() -> pd.DataFrame:
    """A single valid customer row matching the feature contract (the
    real model's 13-column contract — see app/ml/inference/feature_contract.py)."""
    return pd.DataFrame(
        [
            {
                "customer_unique_id": _SAMPLE_ID_1,
                "monetary_value": 245.50,
                "avg_payment_installments": 3,
                "avg_review_score": 2.0,
                "has_bad_review": 1,
                "has_review_comment": 0,
                "avg_product_weight_g": 800.0,
                "freight_ratio": 0.35,
                "avg_delivery_days": 30.0,
                "avg_delivery_delay_days": 6.0,
                "is_delayed_delivery": 1,
                "dominant_product_category_frequency": 0.12,
                "customer_city_state_frequency": 0.05,
                "preferred_payment_type_debit_card": 0,
            }
        ]
    )


@pytest.fixture
def sample_batch_df() -> pd.DataFrame:
    """Multiple valid customer rows (the real model's 13-column contract)."""
    return pd.DataFrame(
        [
            {
                "customer_unique_id": _SAMPLE_ID_1,
                "monetary_value": 245.50,
                "avg_payment_installments": 3,
                "avg_review_score": 2.0,
                "has_bad_review": 1,
                "has_review_comment": 0,
                "avg_product_weight_g": 800.0,
                "freight_ratio": 0.35,
                "avg_delivery_days": 30.0,
                "avg_delivery_delay_days": 6.0,
                "is_delayed_delivery": 1,
                "dominant_product_category_frequency": 0.12,
                "customer_city_state_frequency": 0.05,
                "preferred_payment_type_debit_card": 0,
            },
            {
                "customer_unique_id": _SAMPLE_ID_2,
                "monetary_value": 1840.00,
                "avg_payment_installments": 1,
                "avg_review_score": 4.7,
                "has_bad_review": 0,
                "has_review_comment": 1,
                "avg_product_weight_g": 1500.0,
                "freight_ratio": 0.10,
                "avg_delivery_days": 8.0,
                "avg_delivery_delay_days": -2.0,
                "is_delayed_delivery": 0,
                "dominant_product_category_frequency": 0.30,
                "customer_city_state_frequency": 0.22,
                "preferred_payment_type_debit_card": 1,
            },
            {
                "customer_unique_id": _SAMPLE_ID_3,
                "monetary_value": 610.25,
                "avg_payment_installments": 2,
                "avg_review_score": 3.5,
                "has_bad_review": 0,
                "has_review_comment": 0,
                "avg_product_weight_g": 1100.0,
                "freight_ratio": 0.20,
                "avg_delivery_days": 15.0,
                "avg_delivery_delay_days": 1.0,
                "is_delayed_delivery": 0,
                "dominant_product_category_frequency": 0.18,
                "customer_city_state_frequency": 0.09,
                "preferred_payment_type_debit_card": 0,
            },
        ]
    )


@pytest.fixture(scope="session")
def churn_predictor() -> ChurnPredictor:
    """
    A single ChurnPredictor instance shared across tests in a session (model
    loading + SHAP explainer setup is somewhat expensive, so we don't want
    to repeat it per test).

    Requires MODEL_PATH and MODEL_VERSION to be set in the environment —
    there is no dummy model to fall back to (see app/ml/inference/predict.py).
    """
    return ChurnPredictor(
        model_path=DEFAULT_MODEL_PATH, reason_codes_path=DEFAULT_REASON_CODES_PATH
    )