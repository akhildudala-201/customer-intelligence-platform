"""
smoke_check.py

WHY THIS FILE EXISTS
---------------------
The pytest suite (tests/) is what actually PROVES this part is correct —
it asserts things like "probability is between 0 and 1", "missing a
required feature raises", "SHAP values are JSON-serializable", etc. This
script does NOT replace that: it asserts nothing and can pass its own
eyeball-check while a real pytest test underneath it is failing.

What it's for instead: a fast, visual "does the pipeline even run"
check — load the real model, run the real SHAP explainer, run the real
reason-code logic, on one hardcoded customer, and print what comes out,
so you can look at it without reading pytest's dot-and-traceback output
or wiring up a DB connection. Useful while developing, or to show
someone the pipeline working end-to-end in under a second.

WHAT IT NEEDS
-------------
MODEL_PATH and MODEL_VERSION set (same as everything else in this
package — see inference/predict.py). Nothing DB-related, since it uses
a hardcoded feature row instead of data_access.

RUNNING IT
----------
    python -m app.ml.explainibility_inference.smoke_check
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `app.*` imports resolve
# regardless of whether this script is invoked directly or via -m.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from app.ml.explainibility_inference.inference.predict import ChurnPredictor

# Same shape as tests/inference/conftest.py's sample_customer_df fixture —
# a real Olist-format customer_unique_id and the model's feature contract.
_SAMPLE_CUSTOMER = pd.DataFrame(
    [
        {
            "customer_unique_id": "861eff4711a542e4b93843c6dd7febb0",
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
            "preferred_payment_type_boleto": 0,
            "preferred_payment_type_debit_card": 0,
            "preferred_payment_type_voucher": 0,
        }
    ]
)


def run() -> list[dict]:
    predictor = ChurnPredictor()
    return predictor.predict(_SAMPLE_CUSTOMER)


if __name__ == "__main__":
    print("Loading model and running the pipeline on one sample customer...\n")
    results = run()
    print(json.dumps(results, indent=2))
    print(f"\n{len(results)} record(s) produced. Pipeline runs end-to-end.")
