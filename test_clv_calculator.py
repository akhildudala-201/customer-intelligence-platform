
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.segmentation.customerintelligence.clv_calculator import (
    AVG_ORDER_VALUE_COLUMN,
    CLV_COLUMN,
    DAYS_PER_YEAR,
    FREQUENCY_COLUMN,
    ID_COLUMN,
    LIFESPAN_COLUMN,
    PURCHASE_FREQUENCY_COLUMN,
    SINGLE_ORDER_COLUMN,
    TENURE_DAYS_COLUMN,
    VALUE_TIER_COLUMN,
    ValueTierThresholds,
    _cohort_annual_frequency,
    _cohort_lifespan_years,
    _validate_clv_output,
    compute_customer_clv,
    get_customer_features,
)


def make_features(rows: list[dict]) -> pd.DataFrame:
    """Builds a minimal customer_features_with_labels-shaped DataFrame from
    a list of row dicts, so each test only has to specify the columns it
    actually cares about."""
    return pd.DataFrame(rows)


# A small, hand-computable fixture used across most tests:
#   repeat_1: 4 orders over exactly 1 year   -> 4.0 orders/year
#   repeat_2: 2 orders over exactly 0.5 years -> 4.0 orders/year
#   one_timer: 1 order, tenure_days = 0 (single_order_customer)
BASE_ROWS = [
    {
        ID_COLUMN: "repeat_1",
        FREQUENCY_COLUMN: 4,
        AVG_ORDER_VALUE_COLUMN: 100.0,
        TENURE_DAYS_COLUMN: DAYS_PER_YEAR,  # exactly 1.0 year
        SINGLE_ORDER_COLUMN: 0,
    },
    {
        ID_COLUMN: "repeat_2",
        FREQUENCY_COLUMN: 2,
        AVG_ORDER_VALUE_COLUMN: 50.0,
        TENURE_DAYS_COLUMN: DAYS_PER_YEAR / 2,  # exactly 0.5 years
        SINGLE_ORDER_COLUMN: 0,
    },
    {
        ID_COLUMN: "one_timer",
        FREQUENCY_COLUMN: 1,
        AVG_ORDER_VALUE_COLUMN: 30.0,
        TENURE_DAYS_COLUMN: 0,
        SINGLE_ORDER_COLUMN: 1,
    },
]


class ValueTierThresholdsTests(unittest.TestCase):
    def test_assign_buckets_correctly_including_boundaries(self):
        thresholds = ValueTierThresholds(high_min=100, medium_min=50)
        self.assertEqual(thresholds.assign(150), "High")
        self.assertEqual(thresholds.assign(100), "High")  # boundary is inclusive
        self.assertEqual(thresholds.assign(99.99), "Medium")
        self.assertEqual(thresholds.assign(50), "Medium")  # boundary is inclusive
        self.assertEqual(thresholds.assign(0), "Low")

    def test_high_min_below_medium_min_raises(self):
        with self.assertRaises(ValueError):
            ValueTierThresholds(high_min=10, medium_min=50)

    def test_equal_thresholds_are_allowed(self):
        # high_min == medium_min is a degenerate but valid config (no
        # "Medium" band); should not raise.
        thresholds = ValueTierThresholds(high_min=50, medium_min=50)
        self.assertEqual(thresholds.assign(50), "High")
        self.assertEqual(thresholds.assign(49.99), "Low")


class CohortHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = make_features(BASE_ROWS)

    def test_cohort_lifespan_years_averages_repeat_customers_only(self):
        lifespan = _cohort_lifespan_years(self.features)
        # (1.0 year + 0.5 years) / 2 repeat customers = 0.75 years.
        # The one-timer must NOT be included in this average.
        self.assertAlmostEqual(lifespan, 0.75, places=6)

    def test_cohort_lifespan_years_raises_when_no_repeat_customers_exist(self):
        one_timers_only = self.features[self.features[SINGLE_ORDER_COLUMN] == 1]
        with self.assertRaises(ValueError):
            _cohort_lifespan_years(one_timers_only)

    def test_cohort_annual_frequency_averages_repeat_customers_only(self):
        freq = _cohort_annual_frequency(self.features)
        # Both repeat customers individually work out to 4.0 orders/year,
        # so the cohort average must also be 4.0.
        self.assertAlmostEqual(freq, 4.0, places=6)


class ComputeCustomerClvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = make_features(BASE_ROWS)

    def _row(self, result: pd.DataFrame, customer_id: str) -> pd.Series:
        return result.loc[result[ID_COLUMN] == customer_id].iloc[0]

    def test_repeat_customers_get_their_own_individual_frequency(self):
        result = compute_customer_clv(features=self.features.copy())
        self.assertAlmostEqual(
            self._row(result, "repeat_1")[PURCHASE_FREQUENCY_COLUMN], 4.0, places=3
        )
        self.assertAlmostEqual(
            self._row(result, "repeat_2")[PURCHASE_FREQUENCY_COLUMN], 4.0, places=3
        )

    def test_one_time_buyer_frequency_falls_back_to_cohort_average(self):
        result = compute_customer_clv(features=self.features.copy())
        cohort_freq = _cohort_annual_frequency(self.features)
        one_timer_freq = self._row(result, "one_timer")[PURCHASE_FREQUENCY_COLUMN]
        self.assertAlmostEqual(one_timer_freq, cohort_freq, places=6)
        # And that fallback must be a real, finite number, not NaN/undefined.
        self.assertFalse(np.isnan(one_timer_freq))

    def test_lifespan_is_a_single_shared_value_across_every_row(self):
        result = compute_customer_clv(features=self.features.copy())
        distinct_lifespans = result[LIFESPAN_COLUMN].unique()
        self.assertEqual(len(distinct_lifespans), 1)
        self.assertAlmostEqual(
            distinct_lifespans[0], _cohort_lifespan_years(self.features), places=6
        )

    def test_lifespan_and_frequency_do_not_share_a_denominator(self):
        
        result = compute_customer_clv(features=self.features.copy())
        repeat_1 = self._row(result, "repeat_1")
        naive_revenue_to_date = repeat_1[AVG_ORDER_VALUE_COLUMN] * FREQUENCY_COLUMN_VALUE
        self.assertNotAlmostEqual(repeat_1[CLV_COLUMN], naive_revenue_to_date, places=2)

    def test_clv_equals_aov_times_frequency_times_lifespan(self):
        result = compute_customer_clv(features=self.features.copy())
        for _, row in result.iterrows():
            expected = (
                row[AVG_ORDER_VALUE_COLUMN]
                * row[PURCHASE_FREQUENCY_COLUMN]
                * row[LIFESPAN_COLUMN]
            )
            self.assertAlmostEqual(row[CLV_COLUMN], expected, places=6)

    def test_explicit_lifespan_years_overrides_cohort_average(self):
        result = compute_customer_clv(features=self.features.copy(), lifespan_years=3.0)
        self.assertTrue((result[LIFESPAN_COLUMN] == 3.0).all())

    def test_explicit_lifespan_years_must_be_positive(self):
        with self.assertRaises(ValueError):
            compute_customer_clv(features=self.features.copy(), lifespan_years=0)
        with self.assertRaises(ValueError):
            compute_customer_clv(features=self.features.copy(), lifespan_years=-1.5)

    def test_custom_value_tier_thresholds_are_respected(self):
        thresholds = ValueTierThresholds(high_min=10_000, medium_min=5_000)
        result = compute_customer_clv(
            features=self.features.copy(), value_tier_thresholds=thresholds
        )
        # Every customer's CLV is far below these thresholds, so all rows
        # must land in "Low" -- proves the override is actually used
        # instead of the default percentile split.
        self.assertTrue((result[VALUE_TIER_COLUMN] == "Low").all())

    def test_default_thresholds_produce_only_valid_labels(self):
        result = compute_customer_clv(features=self.features.copy())
        self.assertTrue(
            set(result[VALUE_TIER_COLUMN].unique()).issubset({"High", "Medium", "Low"})
        )

    def test_output_has_exactly_one_row_per_input_customer(self):
        result = compute_customer_clv(features=self.features.copy())
        self.assertEqual(len(result), len(self.features))
        self.assertEqual(result[ID_COLUMN].nunique(), len(result))

    def test_does_not_mutate_the_caller_supplied_dataframe(self):
        original = self.features.copy()
        compute_customer_clv(features=self.features)
        pd.testing.assert_frame_equal(self.features, original)


# Used by test_lifespan_and_frequency_do_not_share_a_denominator above --
# repeat_1's frequency in BASE_ROWS.
FREQUENCY_COLUMN_VALUE = 4


class ValidationTests(unittest.TestCase):
    """Exercises _validate_clv_output's four checks. The first two run it
    indirectly through compute_customer_clv(); the last two call
    _validate_clv_output() directly with a hand-corrupted frame, since
    those failure states can't be reached through normal computation."""

    def setUp(self) -> None:
        self.features = make_features(BASE_ROWS)

    def test_negative_avg_order_value_raises(self):
        bad = self.features.copy()
        bad.loc[0, AVG_ORDER_VALUE_COLUMN] = -50.0
        with self.assertRaises(ValueError):
            compute_customer_clv(features=bad)

    def test_duplicate_customer_id_raises(self):
        bad = pd.concat([self.features, self.features.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            compute_customer_clv(features=bad)

    def test_null_clv_raises(self):
        result = compute_customer_clv(features=self.features.copy())
        result.loc[0, CLV_COLUMN] = np.nan
        with self.assertRaises(ValueError):
            _validate_clv_output(result)

    def test_unexpected_value_tier_label_raises(self):
        result = compute_customer_clv(features=self.features.copy())
        result.loc[0, VALUE_TIER_COLUMN] = "Ultra"
        with self.assertRaises(ValueError):
            _validate_clv_output(result)


class GetCustomerFeaturesTests(unittest.TestCase):
    """Mocks pd.read_sql_table so these never touch a real database."""

    @patch("app.segmentation.customerintelligence.clv_calculator.pd.read_sql_table")
    def test_returns_dataframe_when_table_has_required_columns(self, mock_read_sql_table):
        mock_read_sql_table.return_value = make_features(BASE_ROWS)

        result = get_customer_features()

        self.assertEqual(len(result), len(BASE_ROWS))
        mock_read_sql_table.assert_called_once()

    @patch("app.segmentation.customerintelligence.clv_calculator.pd.read_sql_table")
    def test_empty_table_raises(self, mock_read_sql_table):
        mock_read_sql_table.return_value = pd.DataFrame()
        with self.assertRaises(ValueError):
            get_customer_features()

    @patch("app.segmentation.customerintelligence.clv_calculator.pd.read_sql_table")
    def test_missing_required_column_raises(self, mock_read_sql_table):
        # Missing avg_order_value, tenure_days, single_order_customer.
        mock_read_sql_table.return_value = pd.DataFrame(
            [{ID_COLUMN: "c1", FREQUENCY_COLUMN: 1}]
        )
        with self.assertRaises(ValueError):
            get_customer_features()


if __name__ == "__main__":
    unittest.main()
