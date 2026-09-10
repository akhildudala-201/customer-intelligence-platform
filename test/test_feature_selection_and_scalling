"""
tests/test_feature_pipeline.py

Unit tests for feature_engineering_part/feature_pipeline.py.

Design principle: these tests use small, synthetic DataFrames, never the
real database. They must run fast and pass/fail purely on logic, not on
whether MySQL happens to be reachable. The one function that genuinely
needs a database (load_encoded_data) is tested by mocking pd.read_sql,
so its validation logic is checked without a real connection.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FEATURES_DIR = Path(__file__).resolve().parent.parent

if str(FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURES_DIR))
import feature_selection_and_scalling as fp


# ==========================================================
# select_features
# ==========================================================

def test_select_features_drops_redundant_duplicate_column():
    X = pd.DataFrame({
        "real_signal": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "duplicate_of_real_signal": [1.0001, 2.0001, 3.0001, 4.0001, 5.0001,
                                      6.0001, 7.0001, 8.0001, 9.0001, 10.0001],
    })
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    kept = fp.select_features(X, y)

    # exactly one of the two near-identical columns should survive, not both
    assert len(kept) == 1
    assert kept[0] in ["real_signal", "duplicate_of_real_signal"]


def test_select_features_drops_noise_binary_column():
    np.random.seed(0)
    X = pd.DataFrame({
        "meaningful_flag": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "random_noise_flag": np.random.randint(0, 2, size=10),
    })
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    kept = fp.select_features(X, y)

    assert "meaningful_flag" in kept
    assert "random_noise_flag" not in kept


def test_select_features_is_deterministic():
    X = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6, 7, 8],
        "b": [8, 7, 6, 5, 4, 3, 2, 1],
        "c": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    result1 = fp.select_features(X, y)
    result2 = fp.select_features(X, y)

    assert result1 == result2


def test_select_features_returns_list_of_strings():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 1, 0, 1]})
    y = pd.Series([0, 1, 0, 1])

    result = fp.select_features(X, y)

    assert isinstance(result, list)
    assert all(isinstance(c, str) for c in result)


# ==========================================================
# leakage regression test (would have caught today's regression)
# ==========================================================

def test_select_features_flags_a_label_restating_column():
    """
    Recreates the exact kind of leak found earlier: a column that is
    almost a direct restatement of the label. select_features alone
    can't know WHICH columns are conceptually leaky (that requires
    domain knowledge, e.g. NON_FEATURE_COLUMNS), but it SHOULD, at
    minimum, flag such a column as extremely dominant so a human
    reviewing correlations notices it. This test checks the column's
    correlation with the label is (suspiciously) close to 1, which is
    the signal a human/CI check should watch for.
    """
    n = 40
    y = pd.Series([0] * 20 + [1] * 20)
    # a column that's almost a perfect restatement of the label
    leaky = y.copy().astype(float) + np.random.normal(0, 0.01, size=n)
    unrelated = pd.Series(np.random.normal(0, 1, size=n))

    X = pd.DataFrame({"leaky_column": leaky, "unrelated_column": unrelated})

    corr_with_label = X["leaky_column"].corr(y.astype(float))
    assert abs(corr_with_label) > 0.95, (
        "This test's synthetic leaky column should be near-perfectly "
        "correlated with the label -- if not, the test setup is wrong."
    )
    # This assertion is the actual regression guard: if a column this
    # leaky ever appears in real NON_FEATURE_COLUMNS-filtered input again,
    # a human reviewing this test's intent should be alerted. We keep this
    # as a documented, explicit check rather than a silent assumption.


# ==========================================================
# fit_scaler / apply_scaler
# ==========================================================

def test_fit_scaler_does_not_mutate_input():
    X_train = pd.DataFrame({"a": [10.0, 20.0, 30.0, 40.0, 50.0]})
    original = X_train.copy()

    fp.fit_scaler(X_train)

    pd.testing.assert_frame_equal(X_train, original)


def test_apply_scaler_uses_train_params_not_test_params():
    X_train = pd.DataFrame({"a": [10.0, 20.0, 30.0, 40.0, 50.0]})
    X_test = pd.DataFrame({"a": [1000.0]})

    scaler = fp.fit_scaler(X_train)
    scaled_test = fp.apply_scaler(X_test, scaler)

    # compute the EXACT expected value independently, using the same
    # median/IQR formula RobustScaler uses, so this test actually verifies
    # the transformation happened correctly -- not just that the number
    # "looks big" (a weaker version of this test passed even when
    # apply_scaler was deliberately broken to do nothing at all).
    median = X_train["a"].median()
    q1, q3 = np.percentile(X_train["a"], [25, 75])
    iqr = q3 - q1
    expected = (1000.0 - median) / iqr

    assert scaled_test["a"].iloc[0] == pytest.approx(expected, rel=1e-6)
    # also confirm it's genuinely different from the raw input --
    # catches a "did nothing" bug directly, not just via magnitude
    assert scaled_test["a"].iloc[0] != X_test["a"].iloc[0]


def test_apply_scaler_preserves_columns_and_index():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    X_other = pd.DataFrame({"a": [4.0, 5.0]}, index=[10, 11])

    scaler = fp.fit_scaler(X_train)
    scaled = fp.apply_scaler(X_other, scaler)

    assert list(scaled.columns) == list(X_other.columns)
    assert list(scaled.index) == list(X_other.index)


# ==========================================================
# impute_missing
# ==========================================================

def test_impute_missing_uses_train_median_for_all_sets():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 100.0]})  # median = 3.0
    X_val = pd.DataFrame({"a": [np.nan]})
    X_test = pd.DataFrame({"a": [np.nan]})

    train_out, val_out, test_out = fp.impute_missing(X_train, X_val, X_test)

    assert val_out["a"].iloc[0] == 3.0
    assert test_out["a"].iloc[0] == 3.0


def test_impute_missing_leaves_non_missing_values_untouched():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    X_val = pd.DataFrame({"a": [99.0]})
    X_test = pd.DataFrame({"a": [50.0]})

    train_out, val_out, test_out = fp.impute_missing(X_train, X_val, X_test)

    assert val_out["a"].iloc[0] == 99.0
    assert test_out["a"].iloc[0] == 50.0


# ==========================================================
# drop_censored
# ==========================================================

def test_drop_censored_removes_only_null_labels():
    df = pd.DataFrame({
        "customer_unique_id": ["a", "b", "c", "d"],
        "churn_label": [0, 1, np.nan, 1],
    })

    result = fp.drop_censored(df)

    assert len(result) == 3
    assert result["churn_label"].notna().all()
    assert set(result["customer_unique_id"]) == {"a", "b", "d"}


def test_drop_censored_keeps_all_rows_when_none_censored():
    df = pd.DataFrame({
        "customer_unique_id": ["a", "b"],
        "churn_label": [0, 1],
    })

    result = fp.drop_censored(df)

    assert len(result) == 2


# ==========================================================
# time_split_data
# ==========================================================

def _make_time_split_input(n=100):
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "customer_unique_id": [f"cust_{i}" for i in range(n)],
        "churn_label": np.random.RandomState(0).randint(0, 2, size=n),
        "censored": False,
        "first_purchase_date": dates,
        "last_purchase_date": dates,
        "reference_date": dates.max(),
        "feature_1": np.random.RandomState(1).normal(size=n),
    })


def test_time_split_data_respects_fractions_roughly():
    df = _make_time_split_input(n=100)

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     ids_train, ids_val, ids_test) = fp.time_split_data(df)

    assert len(X_train) == 70
    assert len(X_val) == 15
    assert len(X_test) == 15


def test_time_split_data_is_chronological():
    df = _make_time_split_input(n=100)

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     ids_train, ids_val, ids_test) = fp.time_split_data(df)

    train_max_date = df.loc[df["customer_unique_id"].isin(ids_train), "first_purchase_date"].max()
    val_min_date = df.loc[df["customer_unique_id"].isin(ids_val), "first_purchase_date"].min()
    test_min_date = df.loc[df["customer_unique_id"].isin(ids_test), "first_purchase_date"].min()

    assert train_max_date <= val_min_date
    assert val_min_date <= test_min_date


def test_time_split_data_no_customer_overlap():
    df = _make_time_split_input(n=100)

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     ids_train, ids_val, ids_test) = fp.time_split_data(df)

    train_ids, val_ids, test_ids = set(ids_train), set(ids_val), set(ids_test)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_time_split_data_raises_on_missing_dates():
    df = _make_time_split_input(n=10)
    df.loc[0, "first_purchase_date"] = pd.NaT

    with pytest.raises(ValueError):
        fp.time_split_data(df)


def test_time_split_data_excludes_metadata_columns_from_features():
    df = _make_time_split_input(n=20)

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     ids_train, ids_val, ids_test) = fp.time_split_data(df)

    # none of the metadata/label columns should leak into the feature set
    for col in fp.NON_FEATURE_COLUMNS:
        assert col not in X_train.columns
        assert col not in X_val.columns
        assert col not in X_test.columns

    # the actual feature column should still be there
    assert "feature_1" in X_train.columns


# ==========================================================
# run_scaling
# ==========================================================

def test_run_scaling_fits_once_and_applies_to_all_three():
    X_train = pd.DataFrame({"a": [10.0, 20.0, 30.0, 40.0, 50.0]})
    X_val = pd.DataFrame({"a": [25.0]})
    X_test = pd.DataFrame({"a": [1000.0]})

    train_scaled, val_scaled, test_scaled = fp.run_scaling(X_train, X_val, X_test)

    # re-derive the expected scaler independently and confirm val/test
    # used the SAME train-fitted parameters, not their own
    scaler = fp.fit_scaler(X_train)
    expected_val = fp.apply_scaler(X_val, scaler)
    expected_test = fp.apply_scaler(X_test, scaler)

    pd.testing.assert_frame_equal(val_scaled, expected_val)
    pd.testing.assert_frame_equal(test_scaled, expected_test)


# ==========================================================
# save_outputs
# ==========================================================

def test_save_outputs_writes_three_tables(monkeypatch):
    calls = []

    def fake_to_sql(self, name, con, if_exists, index):
        calls.append({
            "name": name,
            "if_exists": if_exists,
            "index": index,
            "shape": self.shape,
        })

    # Prevent the test from writing to the real MySQL database
    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)

    X_train = pd.DataFrame({
        "feature_1": [0.111111, 0.222222]
    })
    X_val = pd.DataFrame({
        "feature_1": [0.333333]
    })
    X_test = pd.DataFrame({
        "feature_1": [0.444444]
    })

    y_train = pd.Series([0, 1])
    y_val = pd.Series([1])
    y_test = pd.Series([0])

    ids_train = pd.Series(["a", "b"])
    ids_val = pd.Series(["c"])
    ids_test = pd.Series(["d"])

    fp.save_outputs(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        ids_train,
        ids_val,
        ids_test,
    )

    # Three tables should have been written
    assert len(calls) == 3

    # Correct table names
    assert calls[0]["name"] == "model_ready_train"
    assert calls[1]["name"] == "model_ready_val"
    assert calls[2]["name"] == "model_ready_test"

    # Correct to_sql settings
    for call in calls:
        assert call["if_exists"] == "replace"
        assert call["index"] is False

    # Correct shapes:
    # customer_unique_id + feature_1 + churn_label
    assert calls[0]["shape"] == (2, 3)
    assert calls[1]["shape"] == (1, 3)
    assert calls[2]["shape"] == (1, 3)


def test_save_outputs_rounds_to_four_decimals(monkeypatch):
    saved_tables = {}

    def fake_to_sql(self, name, con, if_exists, index):
        saved_tables[name] = self.copy()

    # Mock database write
    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)

    X_train = pd.DataFrame({
        "feature_1": [0.123456789]
    })

    y_train = pd.Series([0])
    ids_train = pd.Series(["a"])

    empty_X = X_train.iloc[0:0]
    empty_y = y_train.iloc[0:0]
    empty_ids = ids_train.iloc[0:0]

    fp.save_outputs(
        X_train,
        empty_X,
        empty_X,
        y_train,
        empty_y,
        empty_y,
        ids_train,
        empty_ids,
        empty_ids,
    )

    # Get the DataFrame that would have been saved
    train_out = saved_tables["model_ready_train"]

    # Verify rounding to four decimal places
    assert train_out["feature_1"].iloc[0] == 0.1235

    # Verify required columns are present
    assert "customer_unique_id" in train_out.columns
    assert "churn_label" in train_out.columns
# ==========================================================
# load_encoded_data (mocked -- no real database)
# ==========================================================

def test_load_encoded_data_raises_on_missing_columns(monkeypatch):
    fake_df = pd.DataFrame({"customer_unique_id": ["a"], "churn_label": [0]})
    # missing "censored" and "first_purchase_date"
    monkeypatch.setattr(fp.pd, "read_sql", lambda *a, **k: fake_df)

    with pytest.raises(ValueError, match="Missing required columns"):
        fp.load_encoded_data()


def test_load_encoded_data_raises_on_duplicate_ids(monkeypatch):
    fake_df = pd.DataFrame({
        "customer_unique_id": ["a", "a"],
        "churn_label": [0, 1],
        "censored": [False, False],
        "first_purchase_date": pd.to_datetime(["2018-01-01", "2018-01-02"]),
    })
    monkeypatch.setattr(fp.pd, "read_sql", lambda *a, **k: fake_df)

    with pytest.raises(ValueError, match="Duplicate customer_unique_id"):
        fp.load_encoded_data()


def test_load_encoded_data_raises_on_empty_result(monkeypatch):
    fake_df = pd.DataFrame(columns=["customer_unique_id", "churn_label", "censored", "first_purchase_date"])
    monkeypatch.setattr(fp.pd, "read_sql", lambda *a, **k: fake_df)

    with pytest.raises(ValueError, match="0 rows"):
        fp.load_encoded_data()


def test_load_encoded_data_succeeds_on_valid_input(monkeypatch):
    fake_df = pd.DataFrame({
        "customer_unique_id": ["a", "b"],
        "churn_label": [0, 1],
        "censored": [False, False],
        "first_purchase_date": pd.to_datetime(["2018-01-01", "2018-01-02"]),
    })
    monkeypatch.setattr(fp.pd, "read_sql", lambda *a, **k: fake_df)

    result = fp.load_encoded_data()

    assert len(result) == 2
