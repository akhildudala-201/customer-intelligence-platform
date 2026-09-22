from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from app.ml.explainibility_inference.inference.predict import ChurnPredictor


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
