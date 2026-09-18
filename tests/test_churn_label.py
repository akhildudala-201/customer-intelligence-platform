import pandas as pd
import numpy as np
import pytest

from app.Features.build_churn_label import load_config


def test_load_config():
    config = load_config()
    assert "reference_date" in config
    assert isinstance(config["reference_date"], pd.Timestamp)
    assert "return_window_days" in config
    assert isinstance(config["return_window_days"], int)


def test_churn_label_logic():
    reference_date = pd.Timestamp("2018-10-17")
    return_window_days = 180

    orders = pd.DataFrame({
        "customer_unique_id": ["u_retained", "u_retained", "u_churned", "u_censored"],
        "order_id": ["o1", "o2", "o3", "o4"],
        "order_purchase_timestamp": [
            pd.Timestamp("2018-01-01"),
            pd.Timestamp("2018-03-01"),  # 59 days gap <= 180 -> retained
            pd.Timestamp("2017-01-01"),  # > 180 days from ref date and no repeat -> churned
            pd.Timestamp("2018-09-01"),  # 46 days from ref date and no repeat -> censored
        ],
    })

    # Sort
    orders = orders.sort_values(["customer_unique_id", "order_purchase_timestamp"]).reset_index(drop=True)

    grouped = orders.groupby("customer_unique_id")
    result = grouped.agg(
        first_order_date=("order_purchase_timestamp", "min"),
        last_order_date=("order_purchase_timestamp", "max"),
        num_valid_orders=("order_id", "nunique"),
    ).reset_index()

    result["days_since_last_order"] = (
        reference_date - result["last_order_date"]
    ).dt.total_seconds() / 86400.0

    orders["prev_order_date"] = orders.groupby("customer_unique_id")["order_purchase_timestamp"].shift(1)
    orders["gap_to_prev"] = (orders["order_purchase_timestamp"] - orders["prev_order_date"]).dt.total_seconds() / 86400.0

    repeat_customers = set(orders.loc[orders["gap_to_prev"] <= return_window_days, "customer_unique_id"])

    result["has_repeat"] = result["customer_unique_id"].isin(repeat_customers)
    result["censored"] = (~result["has_repeat"]) & (result["days_since_last_order"] < return_window_days)
    result["label"] = pd.NA
    result.loc[result["has_repeat"], "label"] = 0
    result.loc[(~result["has_repeat"]) & (result["days_since_last_order"] >= return_window_days), "label"] = 1

    labels_by_user = result.set_index("customer_unique_id")["label"].to_dict()
    censored_by_user = result.set_index("customer_unique_id")["censored"].to_dict()

    assert labels_by_user["u_retained"] == 0
    assert labels_by_user["u_churned"] == 1
    assert pd.isna(labels_by_user["u_censored"])
    assert censored_by_user["u_censored"] is True
    assert censored_by_user["u_retained"] is False
