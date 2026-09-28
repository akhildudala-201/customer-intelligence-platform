
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api.clv_delivery_payments import clv_delivery_endpoints as mod


class BaseApiTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(mod.router)
        self.client = TestClient(app)

    def mock_q(self, *dfs):
        """Patch mod.q to return the given DataFrames in call order."""
        patcher = patch.object(mod, "q", side_effect=list(dfs))
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
class HelperTests(BaseApiTest):
    def test_records_converts_nan_to_none(self):
        df = pd.DataFrame({"a": [1.0, np.nan], "b": ["x", None]})
        out = mod.records(df)
        self.assertEqual(out[0]["a"], 1.0)
        self.assertIsNone(out[1]["a"])

    def test_q_raises_500_when_engine_fails(self):
        with patch.object(mod, "get_engine", side_effect=RuntimeError("db down")):
            with self.assertRaises(HTTPException) as ctx:
                mod.q("SELECT 1")
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("Query failed", ctx.exception.detail)

    def test_endpoint_returns_500_on_db_error(self):
        with patch.object(mod, "get_engine", side_effect=RuntimeError("db down")):
            resp = self.client.get("/clv/summary")
        self.assertEqual(resp.status_code, 500)

    def test_cu_key_uses_unique_id_by_default(self):
        self.assertEqual(mod.CU_KEY, "cu.customer_unique_id")


# --------------------------------------------------------------------------
# CLV endpoints
# --------------------------------------------------------------------------
class ClvEndpointTests(BaseApiTest):
    def test_clv_summary(self):
        stats = pd.DataFrame([{
            "customers": 4, "total_clv": 100.0, "avg_clv": 25.0,
            "max_clv": 40.0, "min_clv": 10.0,
        }])
        vals = pd.DataFrame({"v": [10.0, 20.0, 30.0, 40.0]})
        self.mock_q(stats, vals)

        resp = self.client.get("/clv/summary")
        body = resp.json()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(body["customers"], 4)
        self.assertEqual(body["avg_clv"], 25.0)
        self.assertEqual(body["median_clv"], 25.0)

    def test_clv_summary_empty_table_gives_null_median(self):
        stats = pd.DataFrame([{
            "customers": 0, "total_clv": None, "avg_clv": None,
            "max_clv": None, "min_clv": None,
        }])
        self.mock_q(stats, pd.DataFrame({"v": []}))

        body = self.client.get("/clv/summary").json()
        self.assertIsNone(body["median_clv"])

    def test_customers_by_value_tier_adds_pct_share(self):
        df = pd.DataFrame({
            "value_tier": ["High", "Low"],
            "customers": [30, 70],
            "avg_clv": [500.0, 50.0],
            "total_clv": [15000.0, 3500.0],
        })
        self.mock_q(df)

        body = self.client.get("/customers/byvaluetier").json()
        self.assertEqual([r["value_tier"] for r in body], ["High", "Low"])
        self.assertEqual(body[0]["pct_share"], 30.0)
        self.assertEqual(body[1]["pct_share"], 70.0)

    def test_customer_clv_found(self):
        df = pd.DataFrame([{
            "customer_id": "abc", "clv": 123.4, "value_tier": "High",
            "segment_label": "Champions", "churn_probability": 0.2,
            "risk_tier": "Low",
        }])
        mocked = self.mock_q(df)

        resp = self.client.get("/customers/abc/clv")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["clv"], 123.4)
        self.assertEqual(mocked.call_args_list[0].args[1], {"cid": "abc"})

    def test_customer_clv_not_found_returns_404(self):
        self.mock_q(pd.DataFrame())
        resp = self.client.get("/customers/missing/clv")
        self.assertEqual(resp.status_code, 404)

    def test_clv_distribution_bins_and_counts(self):
        self.mock_q(pd.DataFrame({"v": list(range(1, 101))}))

        body = self.client.get("/clv/distribution?bins=10").json()
        self.assertEqual(len(body), 10)
        self.assertEqual(sum(b["customers"] for b in body), 100)
        self.assertLess(body[0]["bin_start"], body[0]["bin_end"])

    def test_clv_distribution_clips_top_outliers(self):
        self.mock_q(pd.DataFrame({"v": [10.0] * 99 + [1_000_000.0]}))

        body = self.client.get("/clv/distribution?bins=5").json()
        self.assertLess(body[-1]["bin_end"], 1_000_000.0)
        self.assertEqual(sum(b["customers"] for b in body), 100)

    def test_clv_distribution_empty_returns_empty_list(self):
        self.mock_q(pd.DataFrame({"v": []}))
        self.assertEqual(self.client.get("/clv/distribution").json(), [])

    def test_clv_distribution_rejects_invalid_bins(self):
        for bins in (1, 4, 101):
            with self.subTest(bins=bins):
                resp = self.client.get(f"/clv/distribution?bins={bins}")
                self.assertEqual(resp.status_code, 422)

    def test_clv_by_segment(self):
        df = pd.DataFrame({
            "segment_label": ["Champions", "At Risk"],
            "customers": [10, 20],
            "avg_clv": [300.0, 80.0],
            "total_clv": [3000.0, 1600.0],
        })
        self.mock_q(df)

        body = self.client.get("/clv/by-segment").json()
        self.assertEqual(len(body), 2)
        self.assertEqual(body[0]["segment_label"], "Champions")

    def test_clv_top_customers_passes_n(self):
        df = pd.DataFrame({
            "customer_id": ["a", "b"], "clv": [900.0, 800.0],
            "value_tier": ["High", "High"], "segment_label": ["X", "Y"],
            "churn_probability": [0.1, 0.9], "risk_tier": ["Low", "High"],
        })
        mocked = self.mock_q(df)

        resp = self.client.get("/clv/top-customers?n=2")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()), 2)
        self.assertEqual(mocked.call_args_list[0].args[1], {"n": 2})

    def test_clv_top_customers_rejects_invalid_n(self):
        for n in (0, 101):
            with self.subTest(n=n):
                self.assertEqual(
                    self.client.get(f"/clv/top-customers?n={n}").status_code, 422
                )

    def test_clv_revenue_at_risk(self):
        at_risk = pd.DataFrame([{"high_risk_customers": 2, "revenue_at_risk": 200.0}])
        total = pd.DataFrame({"t": [1000.0]})
        mocked = self.mock_q(at_risk, total)

        body = self.client.get("/clv/revenue-at-risk").json()
        self.assertEqual(body["high_risk_customers"], 2)
        self.assertEqual(body["revenue_at_risk"], 200.0)
        self.assertEqual(body["total_clv"], 1000.0)
        self.assertEqual(body["pct_of_total_clv"], 20.0)
        self.assertEqual(
            mocked.call_args_list[0].args[1], {"hr": mod.HR}
        )

    def test_clv_revenue_at_risk_zero_total(self):
        self.mock_q(
            pd.DataFrame([{"high_risk_customers": 0, "revenue_at_risk": 0}]),
            pd.DataFrame({"t": [0]}),
        )
        body = self.client.get("/clv/revenue-at-risk").json()
        self.assertEqual(body["pct_of_total_clv"], 0)


# --------------------------------------------------------------------------
# Customer behaviour, delivery, payments, categories
# --------------------------------------------------------------------------
class DriverEndpointTests(BaseApiTest):
    def test_single_vs_repeat(self):
        df = pd.DataFrame({
            "buyer_type": ["Single purchase", "Repeat purchase"],
            "customers": [90, 10],
        })
        mocked = self.mock_q(df)

        body = self.client.get("/customers/single-vs-repeat").json()
        self.assertEqual(body[0]["pct_share"], 90.0)
        self.assertEqual(body[1]["pct_share"], 10.0)
        # must group by the real customer, not the per-order customer_id
        self.assertIn("customer_unique_id", mocked.call_args_list[0].args[0])

    def test_delivery_performance(self):
        kpi = pd.DataFrame([{
            "delivered_orders": 1000, "avg_delivery_days": 12.5,
            "on_time_pct": 92.0, "late_pct": 8.0,
        }])
        by_status = pd.DataFrame({
            "delivery_status": ["On time", "Late"],
            "orders": [920, 80],
            "avg_delivery_days": [11.0, 25.0],
            "avg_churn_probability": [0.30, 0.55],
        })
        mocked = self.mock_q(kpi, by_status)

        body = self.client.get("/delivery/performance").json()
        self.assertEqual(body["summary"]["on_time_pct"], 92.0)
        self.assertEqual(len(body["by_delivery_status"]), 2)
        self.assertIn("customer_unique_id", mocked.call_args_list[0].args[0])

    def test_installments_distribution(self):
        df = pd.DataFrame({
            "installments": [1, 2, 3],
            "payments": [50, 30, 20],
            "total_value": [5000.0, 3000.0, 2000.0],
            "avg_value": [100.0, 100.0, 100.0],
        })
        self.mock_q(df)

        body = self.client.get("/payments/installments-distribution").json()
        self.assertEqual([r["pct_share"] for r in body], [50.0, 30.0, 20.0])

    def test_category_churn_correlation_passes_params(self):
        df = pd.DataFrame({
            "category": ["bed_bath_table"],
            "customers": [500],
            "avg_churn_probability": [0.62],
            "high_risk_pct": [41.0],
        })
        mocked = self.mock_q(df)

        resp = self.client.get("/products/category-churn-correlation?min_customers=100")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()[0]["category"], "bed_bath_table")
        self.assertEqual(
            mocked.call_args_list[0].args[1],
            {"hr": mod.HR, "minc": 100},
        )

    def test_category_churn_correlation_rejects_invalid_min(self):
        resp = self.client.get("/products/category-churn-correlation?min_customers=0")
        self.assertEqual(resp.status_code, 422)


# --------------------------------------------------------------------------
# Routing sanity check
# --------------------------------------------------------------------------
class RoutingTests(unittest.TestCase):
    def test_all_eleven_routes_registered(self):
        paths = {r.path for r in mod.router.routes}
        expected = {
            "/clv/summary",
            "/customers/byvaluetier",
            "/customers/single-vs-repeat",
            "/customers/{customer_id}/clv",
            "/clv/distribution",
            "/clv/by-segment",
            "/clv/top-customers",
            "/clv/revenue-at-risk",
            "/delivery/performance",
            "/payments/installments-distribution",
            "/products/category-churn-correlation",
        }
        self.assertEqual(paths, expected)


if __name__ == "__main__":
    unittest.main()
