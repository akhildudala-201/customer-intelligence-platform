import os
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import create_engine, text

load_dotenv()

CONFIG = {
    # CLV table (generate_clv_table.py output)
    "clv_table": "customer_clv",
    "clv_customer_id": "customer_unique_id",
    "clv_value": "clv",                # numeric CLV column
    "clv_tier": "value_tier",          # e.g. Low / Medium / High
    # churn predictions table
    "churn_table": "churn_predictions",
    "churn_customer_id": "customer_unique_id",
    "churn_prob": "churn_probability",
    # NOTE: risk_tier lives in customer_segments (not churn_predictions)
    "risk_tier": "risk_tier",
    "high_risk_label": "High",         # matched with LIKE 'High%', so 'High' and 'High Risk' both work
    # segments table
    "seg_table": "customer_segments",
    "seg_customer_id": "customer_unique_id",
    "seg_label": "segment_label",
    # orders / payments / products (Olist-style)
    "orders_table": "orders",
    # orders.customer_id is per-order; the real customer is customers.customer_unique_id
    "customers_table": "customers",
    "cust_order_key": "customer_id",
    "cust_unique_id": "customer_unique_id",
    "orders_customer_id": "customer_id",
    # Which key do customer_clv / churn_predictions / customer_segments use?
    # "unique" -> customer_unique_id, "order" -> customers.customer_id
    "pipeline_key": "unique",
    "orders_order_id": "order_id",
    "orders_status": "order_status",
    "orders_delivered": "order_delivered_customer_date",
    "orders_estimated": "order_estimated_delivery_date",
    "orders_purchased": "order_purchase_timestamp",
    "payments_table": "order_payments",
    "pay_order_id": "order_id",
    "pay_installments": "payment_installments",
    "pay_value": "payment_value",
    "items_table": "order_items",
    "items_order_id": "order_id",
    "items_product_id": "product_id",
    "products_table": "products",
    "products_product_id": "product_id",
    "products_category": "product_category_name",
}
C = CONFIG

# Column (on the customers table aliased `cu`) that matches the pipeline tables' customer key
CU_KEY = f"cu.{C['cust_unique_id'] if C['pipeline_key'] == 'unique' else C['cust_order_key']}"

HR = f"{C['high_risk_label']}%"  # LIKE pattern for the high-risk tier

# --------------------------------------------------------------------------
# DB helpers
# --------------------------------------------------------------------------
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        url = os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL is not set in .env")
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def q(sql: str, params: Optional[dict] = None) -> pd.DataFrame:
    try:
        with get_engine().connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


def records(df: pd.DataFrame) -> list:
    df = df.replace({np.nan: None})
    return df.to_dict(orient="records")


router = APIRouter(tags=["P4 - CLV, Delivery & Payments"])

# --------------------------------------------------------------------------
# CLV
# --------------------------------------------------------------------------


@router.get("/clv/summary")
def clv_summary():
    df = q(f"""
        SELECT COUNT(*) AS customers,
               SUM({C['clv_value']}) AS total_clv,
               AVG({C['clv_value']}) AS avg_clv,
               MAX({C['clv_value']}) AS max_clv,
               MIN({C['clv_value']}) AS min_clv
        FROM {C['clv_table']}
    """)
    vals = q(f"SELECT {C['clv_value']} AS v FROM {C['clv_table']}")["v"].dropna()
    out = records(df)[0]
    out["median_clv"] = float(vals.median()) if len(vals) else None
    return out


@router.get("/customers/byvaluetier")
def customers_by_value_tier():
    df = q(f"""
        SELECT {C['clv_tier']} AS value_tier,
               COUNT(*) AS customers,
               AVG({C['clv_value']}) AS avg_clv,
               SUM({C['clv_value']}) AS total_clv
        FROM {C['clv_table']}
        GROUP BY {C['clv_tier']}
        ORDER BY total_clv DESC
    """)
    total = df["customers"].sum()
    df["pct_share"] = (df["customers"] / total * 100).round(2) if total else 0
    return records(df)


@router.get("/customers/single-vs-repeat")
def single_vs_repeat():
    df = q(f"""
        SELECT CASE WHEN n_orders = 1 THEN 'Single purchase'
                    ELSE 'Repeat purchase' END AS buyer_type,
               COUNT(*) AS customers
        FROM (
            SELECT {CU_KEY} AS cid,
                   COUNT(DISTINCT o.{C['orders_order_id']}) AS n_orders
            FROM {C['orders_table']} o
            JOIN {C['customers_table']} cu
              ON cu.{C['cust_order_key']} = o.{C['orders_customer_id']}
            WHERE o.{C['orders_status']} = 'delivered'
            GROUP BY {CU_KEY}
        ) t
        GROUP BY buyer_type
    """)
    total = df["customers"].sum()
    df["pct_share"] = (df["customers"] / total * 100).round(2) if total else 0
    return records(df)


@router.get("/customers/{customer_id}/clv")
def customer_clv(customer_id: str):
    df = q(f"""
        SELECT c.*, s.{C['seg_label']} AS segment_label,
               p.{C['churn_prob']} AS churn_probability,
               s.{C['risk_tier']} AS risk_tier
        FROM {C['clv_table']} c
        LEFT JOIN {C['seg_table']} s ON s.{C['seg_customer_id']} = c.{C['clv_customer_id']}
        LEFT JOIN {C['churn_table']} p ON p.{C['churn_customer_id']} = c.{C['clv_customer_id']}
        WHERE c.{C['clv_customer_id']} = :cid
        LIMIT 1
    """, {"cid": customer_id})
    if df.empty:
        raise HTTPException(status_code=404, detail="Customer not found")
    return records(df)[0]


@router.get("/clv/distribution")
def clv_distribution(bins: int = Query(20, ge=5, le=100)):
    vals = q(f"SELECT {C['clv_value']} AS v FROM {C['clv_table']}")["v"].dropna()
    if vals.empty:
        return []
    # clip top 1% so a few outliers don't flatten the histogram
    upper = float(vals.quantile(0.99))
    counts, edges = np.histogram(vals.clip(upper=upper), bins=bins)
    return [
        {
            "bin_start": round(float(edges[i]), 2),
            "bin_end": round(float(edges[i + 1]), 2),
            "customers": int(counts[i]),
        }
        for i in range(len(counts))
    ]


@router.get("/clv/by-segment")
def clv_by_segment():
    df = q(f"""
        SELECT s.{C['seg_label']} AS segment_label,
               COUNT(*) AS customers,
               AVG(c.{C['clv_value']}) AS avg_clv,
               SUM(c.{C['clv_value']}) AS total_clv
        FROM {C['clv_table']} c
        JOIN {C['seg_table']} s ON s.{C['seg_customer_id']} = c.{C['clv_customer_id']}
        GROUP BY s.{C['seg_label']}
        ORDER BY avg_clv DESC
    """)
    return records(df)


@router.get("/clv/top-customers")
def clv_top_customers(n: int = Query(10, ge=1, le=100)):
    df = q(f"""
        SELECT c.{C['clv_customer_id']} AS customer_id,
               c.{C['clv_value']} AS clv,
               c.{C['clv_tier']} AS value_tier,
               s.{C['seg_label']} AS segment_label,
               p.{C['churn_prob']} AS churn_probability,
               s.{C['risk_tier']} AS risk_tier
        FROM {C['clv_table']} c
        LEFT JOIN {C['seg_table']} s ON s.{C['seg_customer_id']} = c.{C['clv_customer_id']}
        LEFT JOIN {C['churn_table']} p ON p.{C['churn_customer_id']} = c.{C['clv_customer_id']}
        ORDER BY c.{C['clv_value']} DESC
        LIMIT :n
    """, {"n": n})
    return records(df)


@router.get("/clv/revenue-at-risk")
def clv_revenue_at_risk():
    df = q(f"""
        SELECT COUNT(*) AS high_risk_customers,
               COALESCE(SUM(c.{C['clv_value']}), 0) AS revenue_at_risk
        FROM {C['clv_table']} c
        JOIN {C['seg_table']} s ON s.{C['seg_customer_id']} = c.{C['clv_customer_id']}
        WHERE s.{C['risk_tier']} LIKE :hr
    """, {"hr": HR})
    total = q(f"SELECT COALESCE(SUM({C['clv_value']}), 0) AS t FROM {C['clv_table']}")["t"].iloc[0]
    out = records(df)[0]
    out["total_clv"] = float(total)
    out["pct_of_total_clv"] = (
        round(float(out["revenue_at_risk"]) / float(total) * 100, 2) if total else 0
    )
    return out


# --------------------------------------------------------------------------
# Delivery and payment drivers
# --------------------------------------------------------------------------


@router.get("/delivery/performance")
def delivery_performance():
    """Overall delivery KPIs plus churn probability for on-time vs late customers."""
    base = f"""
        SELECT {CU_KEY} AS customer_id,
               DATEDIFF(o.{C['orders_delivered']}, o.{C['orders_purchased']}) AS delivery_days,
               CASE WHEN o.{C['orders_delivered']} > o.{C['orders_estimated']}
                    THEN 'Late' ELSE 'On time' END AS delivery_status
        FROM {C['orders_table']} o
        JOIN {C['customers_table']} cu
          ON cu.{C['cust_order_key']} = o.{C['orders_customer_id']}
        WHERE o.{C['orders_status']} = 'delivered'
          AND o.{C['orders_delivered']} IS NOT NULL
    """
    kpi = q(f"""
        SELECT COUNT(*) AS delivered_orders,
               AVG(delivery_days) AS avg_delivery_days,
               100 * AVG(delivery_status = 'On time') AS on_time_pct,
               100 * AVG(delivery_status = 'Late') AS late_pct
        FROM ({base}) d
    """)
    by_status = q(f"""
        SELECT d.delivery_status,
               COUNT(*) AS orders,
               AVG(d.delivery_days) AS avg_delivery_days,
               AVG(p.{C['churn_prob']}) AS avg_churn_probability
        FROM ({base}) d
        LEFT JOIN {C['churn_table']} p ON p.{C['churn_customer_id']} = d.customer_id
        GROUP BY d.delivery_status
    """)
    return {"summary": records(kpi)[0], "by_delivery_status": records(by_status)}


@router.get("/payments/installments-distribution")
def installments_distribution():
    df = q(f"""
        SELECT {C['pay_installments']} AS installments,
               COUNT(*) AS payments,
               SUM({C['pay_value']}) AS total_value,
               AVG({C['pay_value']}) AS avg_value
        FROM {C['payments_table']}
        GROUP BY {C['pay_installments']}
        ORDER BY installments
    """)
    total = df["payments"].sum()
    df["pct_share"] = (df["payments"] / total * 100).round(2) if total else 0
    return records(df)


@router.get("/products/category-churn-correlation")
def category_churn_correlation(min_customers: int = Query(30, ge=1)):
    """Avg churn probability and high-risk share per product category."""
    df = q(f"""
        SELECT cc.category,
               COUNT(*) AS customers,
               AVG(p.{C['churn_prob']}) AS avg_churn_probability,
               100 * AVG(s.{C['risk_tier']} LIKE :hr) AS high_risk_pct
        FROM (
            SELECT DISTINCT {CU_KEY} AS customer_id,
                   pr.{C['products_category']} AS category
            FROM {C['orders_table']} o
            JOIN {C['customers_table']} cu
              ON cu.{C['cust_order_key']} = o.{C['orders_customer_id']}
            JOIN {C['items_table']} i ON i.{C['items_order_id']} = o.{C['orders_order_id']}
            JOIN {C['products_table']} pr ON pr.{C['products_product_id']} = i.{C['items_product_id']}
            WHERE pr.{C['products_category']} IS NOT NULL
        ) cc
        JOIN {C['churn_table']} p ON p.{C['churn_customer_id']} = cc.customer_id
        JOIN {C['seg_table']} s ON s.{C['seg_customer_id']} = cc.customer_id
        GROUP BY cc.category
        HAVING customers >= :minc
        ORDER BY avg_churn_probability DESC
    """, {"hr": HR, "minc": min_customers})
    return records(df)
