import pandas as pd

from app.segmentation.customer_analytics.cohort_analysis import (
    TABLE_NAME,
    build_scoped_input,
    calculate_cohort_analysis,
)


def test_scoped_input_uses_unique_customer_and_sums_order_payments():
    customers = pd.DataFrame(
        {
            "customer_id": ["order-customer-1", "order-customer-2", "order-customer-3"],
            "customer_unique_id": ["person-1", "person-1", "person-2"],
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "customer_id": ["order-customer-1", "order-customer-2", "order-customer-3"],
            "order_purchase_timestamp": [
                "2017-01-10 10:00:00",
                "2017-03-11 10:00:00",
                "2017-02-05 10:00:00",
            ],
        }
    )
    payments = pd.DataFrame(
        {"order_id": ["o1", "o1", "o2", "o3"], "payment_value": [10, 2.5, 20, 8]}
    )

    result = build_scoped_input(customers, orders, payments)

    assert list(result.columns) == [
        "customer_unique_id",
        "first_purchase_date",
        "order_purchase_timestamp",
        "payment_value",
        "total_orders",
    ]
    assert result.loc[result["customer_unique_id"] == "person-1", "payment_value"].tolist() == [
        12.5,
        20.0,
    ]
    assert result.loc[result["customer_unique_id"] == "person-1", "total_orders"].eq(2).all()


def test_cohort_metrics_are_relative_and_include_m0_through_m3():
    scoped = pd.DataFrame(
        {
            "customer_unique_id": ["a", "a", "b", "c", "c", "d"],
            "first_purchase_date": pd.to_datetime(
                ["2017-01-01", "2017-01-01", "2017-01-15", "2017-02-01", "2017-02-01", "2017-02-10"]
            ),
            "order_purchase_timestamp": pd.to_datetime(
                ["2017-01-01", "2017-03-01", "2017-01-15", "2017-02-01", "2017-05-01", "2017-02-10"]
            ),
            "payment_value": [10, 30, 20, 5, 15, 7],
            "total_orders": [2, 2, 1, 2, 2, 1],
        }
    )

    result = calculate_cohort_analysis(scoped, generated_date="2018-01-01")

    assert TABLE_NAME == "customer_cohort_analysis"
    jan = result[result["cohort"] == "Jan 2017"].set_index("relative_month")
    assert jan.loc["M0", "retention_rate (%)"] == 100.0
    assert jan.loc["M1", "retention_rate (%)"] == 0.0
    assert jan.loc["M2", "retention_rate (%)"] == 50.0
    assert jan.loc["M2", "average_clv"] == 30.0
    assert jan.loc["M2", "repeat_purchase_rate (%)"] == 50.0

    feb = result[result["cohort"] == "Feb 2017"].set_index("relative_month")
    assert feb.loc["M0", "retention_rate (%)"] == 100.0
    assert feb.loc["M3", "retention_rate (%)"] == 50.0


def test_min_orders_is_local_to_cohort_output():
    scoped = pd.DataFrame(
        {
            "customer_unique_id": ["one-time", "repeat", "repeat"],
            "first_purchase_date": pd.to_datetime(
                ["2017-01-01", "2017-01-02", "2017-01-02"]
            ),
            "order_purchase_timestamp": pd.to_datetime(
                ["2017-01-01", "2017-01-02", "2017-03-02"]
            ),
            "payment_value": [5, 10, 20],
            "total_orders": [1, 2, 2],
        }
    )

    result = calculate_cohort_analysis(scoped, min_orders=2)

    assert set(result["cohort"]) == {"Jan 2017"}
    assert result.iloc[0]["repeat_purchase_rate (%)"] == 100.0
