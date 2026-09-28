import pandas as pd
import pytest
from sqlalchemy import create_engine

from app.segmentation.customer_analytics.cohort_analysis import (
    TABLE_NAME,
    build_scoped_input,
    calculate_cohort_analysis,
    load_scoped_input,
    write_cohort_analysis,
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


def test_scoped_input_handles_missing_payment_rows_and_invalid_values():
    customers = pd.DataFrame(
        {"customer_id": ["c1"], "customer_unique_id": ["person-1"]}
    )
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2"],
            "customer_id": ["c1", "c1"],
            "order_purchase_timestamp": ["2017-01-01", "2017-02-01"],
        }
    )
    payments = pd.DataFrame(
        {"order_id": ["o1", "o1"], "payment_value": ["10.5", "invalid"]}
    )

    result = build_scoped_input(customers, orders, payments)

    assert result["payment_value"].tolist() == [10.5, 0.0]
    assert result["first_purchase_date"].eq(pd.Timestamp("2017-01-01")).all()
    assert result["total_orders"].eq(2).all()


def test_scoped_input_rejects_missing_source_columns():
    with pytest.raises(ValueError, match="orders is missing required column"):
        build_scoped_input(
            pd.DataFrame(
                {"customer_id": ["c1"], "customer_unique_id": ["person-1"]}
            ),
            pd.DataFrame({"order_id": ["o1"], "customer_id": ["c1"]}),
            pd.DataFrame({"order_id": ["o1"], "payment_value": [10]}),
        )


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
    assert jan.loc["M0", "cumulative_average_revenue"] == 15.0
    assert jan.loc["M2", "cumulative_average_revenue"] == 30.0
    assert jan.loc["M0", "repeat_purchase_rate (%)"] == 0.0
    assert jan.loc["M2", "repeat_purchase_rate (%)"] == 50.0

    feb = result[result["cohort"] == "Feb 2017"].set_index("relative_month")
    assert feb.loc["M0", "retention_rate (%)"] == 100.0
    assert feb.loc["M3", "retention_rate (%)"] == 50.0
    assert list(result.columns) == [
        "cohort",
        "relative_month",
        "retention_rate (%)",
        "repeat_purchase_rate (%)",
        "cumulative_repeat_purchase_rate (%)",
        "final_churn_rate (%)",
        "monthly_churn_rate (%)",
        "cumulative_churn_rate (%)",
        "cumulative_average_revenue",
        "generated_date",
    ]
    assert result["generated_date"].eq(pd.Timestamp("2018-01-01").date()).all()


def test_cohort_metrics_follow_specified_clv_and_repeat_formulas():
    scoped = pd.DataFrame(
        {
            "customer_unique_id": ["a", "a", "b"],
            "first_purchase_date": pd.to_datetime(
                ["2017-01-01", "2017-01-01", "2017-01-15"]
            ),
            "order_purchase_timestamp": pd.to_datetime(
                ["2017-01-01", "2017-02-01", "2017-01-15"]
            ),
            "payment_value": [100, 50, 25],
            "total_orders": [2, 2, 1],
        }
    )

    result = calculate_cohort_analysis(scoped, generated_date="2018-01-01")

    jan = result[result["cohort"] == "Jan 2017"]
    assert jan.set_index("relative_month")[
        "cumulative_average_revenue"
    ].to_dict() == {
        "M0": 62.5,
        "M1": 87.5,
    }
    assert jan.set_index("relative_month")["repeat_purchase_rate (%)"].to_dict() == {
        "M0": 0.0,
        "M1": 50.0,
    }
    assert jan.set_index("relative_month")[
        "cumulative_repeat_purchase_rate (%)"
    ].to_dict() == {"M0": 0.0, "M1": 50.0}
    assert jan.set_index("relative_month").loc["M1", "retention_rate (%)"] == 50.0


def test_cumulative_average_revenue_carries_forward_when_month_has_no_revenue():
    scoped = pd.DataFrame(
        {
            "customer_unique_id": ["a", "a", "b"],
            "first_purchase_date": pd.to_datetime(
                ["2017-01-01", "2017-01-01", "2017-01-02"]
            ),
            "order_purchase_timestamp": pd.to_datetime(
                ["2017-01-01", "2017-03-01", "2017-04-01"]
            ),
            "payment_value": [100, 50, 25],
            "total_orders": [2, 2, 1],
        }
    )

    result = calculate_cohort_analysis(scoped, generated_date="2018-01-01")
    jan = result[result["cohort"] == "Jan 2017"].set_index("relative_month")

    assert jan.loc["M0", "cumulative_average_revenue"] == 50.0
    assert jan.loc["M1", "cumulative_average_revenue"] == 50.0
    assert jan.loc["M2", "cumulative_average_revenue"] == 75.0


def test_cohort_churn_rate_uses_project_labels_and_excludes_censored_customers():
    scoped = pd.DataFrame(
        {
            "customer_unique_id": ["churned", "retained", "censored"],
            "first_purchase_date": pd.to_datetime(
                ["2017-01-01", "2017-01-02", "2017-01-03"]
            ),
            "order_purchase_timestamp": pd.to_datetime(
                ["2017-01-01", "2017-01-02", "2017-01-03"]
            ),
            "payment_value": [10, 20, 30],
            "total_orders": [1, 1, 1],
            "label": [1, 0, 1],
            "censored": [False, False, True],
        }
    )

    result = calculate_cohort_analysis(scoped, generated_date="2018-01-01")

    jan = result[result["cohort"] == "Jan 2017"]
    assert jan["final_churn_rate (%)"].eq(50.0).all()
    assert jan["cumulative_average_revenue"].eq(15.0).all()


def test_churn_rate_excludes_censored_customers_even_when_retention_keeps_them():
    scoped = pd.DataFrame(
        {
            "customer_unique_id": ["churned", "retained", "censored"],
            "first_purchase_date": pd.to_datetime(
                ["2017-01-01", "2017-01-02", "2017-01-03"]
            ),
            "order_purchase_timestamp": pd.to_datetime(
                ["2017-01-01", "2017-01-02", "2017-01-03"]
            ),
            "payment_value": [10, 20, 30],
            "total_orders": [1, 1, 1],
            "label": [1, 0, 1],
            "censored": [False, False, True],
        }
    )

    result = calculate_cohort_analysis(
        scoped, exclude_censored=False, generated_date="2018-01-01"
    )

    assert result["final_churn_rate (%)"].eq(50.0).all()


def test_monthly_churn_rate_uses_canonical_180_day_inactivity_window():
    scoped = pd.DataFrame(
        {
            "customer_unique_id": [
                "early",
                "early",
                "late",
                "late",
                "retained",
                "retained",
            ],
            "first_purchase_date": pd.to_datetime(
                [
                    "2017-01-01",
                    "2017-01-01",
                    "2017-01-02",
                    "2017-01-02",
                    "2017-01-03",
                    "2017-01-03",
                ]
            ),
            "order_purchase_timestamp": pd.to_datetime(
                [
                    "2017-01-01",
                    "2017-01-31",
                    "2017-01-02",
                    "2017-02-15",
                    "2017-01-03",
                    "2017-03-03",
                ]
            ),
            "payment_value": [10, 5, 10, 5, 10, 5],
            "total_orders": [2, 2, 1, 1, 2, 2],
            "label": [1, 1, 1, 1, 0, 0],
            "censored": [False, False, False, False, False, False],
        }
    )

    result = calculate_cohort_analysis(scoped, generated_date="2018-01-01")
    jan = result[result["cohort"] == "Jan 2017"].set_index("relative_month")

    assert jan.loc["M0", "monthly_churn_rate (%)"] == 0.0
    assert jan.loc["M6", "monthly_churn_rate (%)"] == 33.33
    assert jan.loc["M7", "monthly_churn_rate (%)"] == 50.0
    assert jan.loc["M6", "cumulative_churn_rate (%)"] == 33.33
    assert jan.loc["M7", "cumulative_churn_rate (%)"] == 66.67
    assert jan["final_churn_rate (%)"].eq(66.67).all()


def test_scoped_input_preserves_joined_churn_labels():
    customers = pd.DataFrame(
        {"customer_id": ["c1"], "customer_unique_id": ["person-1"]}
    )
    orders = pd.DataFrame(
        {
            "order_id": ["o1"],
            "customer_id": ["c1"],
            "order_purchase_timestamp": ["2017-01-01"],
        }
    )
    payments = pd.DataFrame({"order_id": ["o1"], "payment_value": [10]})
    churn_labels = pd.DataFrame(
        {
            "customer_unique_id": ["person-1"],
            "label": [1],
            "censored": [False],
        }
    )

    result = build_scoped_input(customers, orders, payments, churn_labels)

    assert result.loc[0, "label"] == 1
    assert result.loc[0, "censored"] == False


def test_invalid_order_threshold_and_empty_eligible_input():
    scoped = pd.DataFrame(
        {
            "customer_unique_id": ["one-time"],
            "first_purchase_date": pd.to_datetime(["2017-01-01"]),
            "order_purchase_timestamp": pd.to_datetime(["2017-01-01"]),
            "payment_value": [10],
            "total_orders": [1],
        }
    )

    with pytest.raises(ValueError, match="min_orders"):
        calculate_cohort_analysis(scoped, min_orders=0)

    result = calculate_cohort_analysis(scoped, min_orders=2)
    assert result.empty
    assert list(result.columns) == [
        "cohort",
        "relative_month",
        "retention_rate (%)",
        "repeat_purchase_rate (%)",
        "cumulative_repeat_purchase_rate (%)",
        "final_churn_rate (%)",
        "monthly_churn_rate (%)",
        "cumulative_churn_rate (%)",
        "cumulative_average_revenue",
        "generated_date",
    ]


def test_invalid_dates_are_excluded_from_cohort_output():
    scoped = pd.DataFrame(
        {
            "customer_unique_id": ["valid", "invalid"],
            "first_purchase_date": ["2017-01-01", "not-a-date"],
            "order_purchase_timestamp": ["2017-01-01", "2017-01-01"],
            "payment_value": [10, 20],
            "total_orders": [1, 1],
        }
    )

    result = calculate_cohort_analysis(scoped, generated_date="2018-01-01")

    assert set(result["cohort"]) == {"Jan 2017"}
    assert result["cumulative_average_revenue"].eq(10.0).all()


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
    assert result.iloc[0]["repeat_purchase_rate (%)"] == 0.0


def test_load_scoped_input_reads_database_tables_and_writer_persists_result():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    try:
        pd.DataFrame(
            {
                "customer_id": ["c1"],
                "customer_unique_id": ["person-1"],
            }
        ).to_sql("customers", engine, index=False)
        pd.DataFrame(
            {
                "order_id": ["o1"],
                "customer_id": ["c1"],
                "order_purchase_timestamp": ["2017-01-01"],
            }
        ).to_sql("orders", engine, index=False)
        pd.DataFrame(
            {"order_id": ["o1"], "payment_value": [42.0]}
        ).to_sql("order_payments", engine, index=False)
        pd.DataFrame(
            {
                "customer_unique_id": ["person-1"],
                "label": [0],
                "censored": [False],
            }
        ).to_sql("customer_churn_labels", engine, index=False)

        scoped = load_scoped_input(connection=engine)
        result = calculate_cohort_analysis(scoped, generated_date="2018-01-01")
        write_cohort_analysis(result, engine)

        stored = pd.read_sql_table(TABLE_NAME, engine)
        assert len(scoped) == 1
        pd.testing.assert_frame_equal(
            stored.assign(
                generated_date=pd.to_datetime(stored["generated_date"]).dt.date
            ),
            result,
            check_dtype=False,
        )
    finally:
        engine.dispose()
