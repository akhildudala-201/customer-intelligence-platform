"""
Unit tests for build_churn_label.py

Run with:
    pytest test_build_churn_label.py -v

These tests mock out the database engine and file system so they run
without needing a real MySQL connection or a real label_config.yaml.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import yaml

# Adjust this import to match wherever build_churn_label.py actually lives
# in your project, e.g.:
#   from app.Features.build_churn_label import (...)
from app.Features.build_churn_label import (
    load_config,
    fetch_valid_orders,
    build_labels,
    save_labels_to_db,
    CONFIG_PATH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_config_yaml():
    return {
        "reference_date": "2018-10-01",
        "return_window_days": 180,
        "exclude_censored_customers": False,
    }


@pytest.fixture
def sample_orders_df():
    """
    Small hand-crafted dataset covering all three label outcomes:

    - cust_retained : two orders 30 days apart -> Retained (0)
    - cust_churned   : one order, 200 days before reference_date -> Churned (1)
    - cust_censored  : one order, 10 days before reference_date -> Censored
    """
    return pd.DataFrame(
        {
            "customer_unique_id": [
                "cust_retained",
                "cust_retained",
                "cust_churned",
                "cust_censored",
            ],
            "order_id": ["o1", "o2", "o3", "o4"],
            "order_status": ["delivered", "delivered", "delivered", "delivered"],
            "order_purchase_timestamp": pd.to_datetime(
                [
                    "2018-08-01",
                    "2018-08-31",  # 30 days after o1 -> qualifies as repeat
                    "2018-03-15",  # ~200 days before reference_date (2018-10-01)
                    "2018-09-21",  # ~10 days before reference_date
                ]
            ),
        }
    )


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:

    def test_load_config_success(self, valid_config_yaml, tmp_path, monkeypatch):
        config_file = tmp_path / "label_config.yaml"
        config_file.write_text(yaml.dump(valid_config_yaml), encoding="utf-8")

        monkeypatch.setattr(
            "app.Features.build_churn_label.CONFIG_PATH", config_file
        )

        config = load_config()

        assert config["return_window_days"] == 180
        assert config["exclude_censored_customers"] is False
        assert isinstance(config["reference_date"], pd.Timestamp)
        assert config["reference_date"] == pd.Timestamp("2018-10-01")

    def test_load_config_missing_file_raises(self, tmp_path, monkeypatch):
        missing_path = tmp_path / "does_not_exist.yaml"
        monkeypatch.setattr(
            "app.Features.build_churn_label.CONFIG_PATH", missing_path
        )

        with pytest.raises(FileNotFoundError):
            load_config()

    def test_load_config_missing_reference_date_raises(self, tmp_path, monkeypatch):
        config_file = tmp_path / "label_config.yaml"
        config_file.write_text(
            yaml.dump({"return_window_days": 90}), encoding="utf-8"
        )

        monkeypatch.setattr(
            "app.Features.build_churn_label.CONFIG_PATH", config_file
        )

        with pytest.raises(ValueError, match="reference_date"):
            load_config()

    def test_load_config_defaults_when_optional_keys_missing(
        self, tmp_path, monkeypatch
    ):
        config_file = tmp_path / "label_config.yaml"
        config_file.write_text(
            yaml.dump({"reference_date": "2018-10-01"}), encoding="utf-8"
        )

        monkeypatch.setattr(
            "app.Features.build_churn_label.CONFIG_PATH", config_file
        )

        config = load_config()

        # return_window_days / exclude_censored_customers are read with
        # .get(...) elsewhere, so load_config itself shouldn't require them.
        assert "reference_date" in config


# ---------------------------------------------------------------------------
# fetch_valid_orders
# ---------------------------------------------------------------------------

class TestFetchValidOrders:

    @patch("app.Features.build_churn_label.pd.read_sql")
    def test_fetch_valid_orders_returns_dataframe(self, mock_read_sql, sample_orders_df):
        mock_read_sql.return_value = sample_orders_df.copy()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = MagicMock()

        result = fetch_valid_orders(
            mock_engine,
            reference_date=pd.Timestamp("2018-10-01"),
            excluded_statuses=[],
        )

        assert not result.empty
        assert "order_purchase_timestamp" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(
            result["order_purchase_timestamp"]
        )

    @patch("app.Features.build_churn_label.pd.read_sql")
    def test_fetch_valid_orders_empty_result(self, mock_read_sql):
        mock_read_sql.return_value = pd.DataFrame(
            columns=[
                "customer_unique_id",
                "order_id",
                "order_status",
                "order_purchase_timestamp",
            ]
        )
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = MagicMock()

        result = fetch_valid_orders(
            mock_engine,
            reference_date=pd.Timestamp("2018-10-01"),
            excluded_statuses=[],
        )

        assert result.empty

    @patch("app.Features.build_churn_label.pd.read_sql")
    def test_fetch_valid_orders_passes_placeholder_when_no_excluded_statuses(
        self, mock_read_sql, sample_orders_df
    ):
        mock_read_sql.return_value = sample_orders_df.copy()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = MagicMock()

        fetch_valid_orders(
            mock_engine,
            reference_date=pd.Timestamp("2018-10-01"),
            excluded_statuses=[],
        )

        _, kwargs = mock_read_sql.call_args
        assert kwargs["params"]["excluded_statuses"] == (
            "__NO_EXCLUDED_STATUS__",
        )

    @patch("app.Features.build_churn_label.pd.read_sql")
    def test_fetch_valid_orders_passes_given_excluded_statuses(
        self, mock_read_sql, sample_orders_df
    ):
        mock_read_sql.return_value = sample_orders_df.copy()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = MagicMock()

        fetch_valid_orders(
            mock_engine,
            reference_date=pd.Timestamp("2018-10-01"),
            excluded_statuses=["canceled", "unavailable"],
        )

        _, kwargs = mock_read_sql.call_args
        assert kwargs["params"]["excluded_statuses"] == (
            "canceled",
            "unavailable",
        )


# ---------------------------------------------------------------------------
# build_labels  (the core labeling logic)
# ---------------------------------------------------------------------------

class TestBuildLabels:

    @patch("app.Features.build_churn_label.fetch_valid_orders")
    @patch("app.Features.build_churn_label.load_config")
    def test_retained_customer_labeled_zero(
        self, mock_load_config, mock_fetch, sample_orders_df, valid_config_yaml
    ):
        cfg = dict(valid_config_yaml)
        cfg["reference_date"] = pd.Timestamp("2018-10-01")
        mock_load_config.return_value = cfg
        mock_fetch.return_value = sample_orders_df.copy()

        result, reference_date = build_labels()

        row = result[result["customer_unique_id"] == "cust_retained"].iloc[0]
        assert row["label"] == 0
        assert row["censored"] == False  # noqa: E712
        assert row["num_valid_orders"] == 2

    @patch("app.Features.build_churn_label.fetch_valid_orders")
    @patch("app.Features.build_churn_label.load_config")
    def test_churned_customer_labeled_one(
        self, mock_load_config, mock_fetch, sample_orders_df, valid_config_yaml
    ):
        cfg = dict(valid_config_yaml)
        cfg["reference_date"] = pd.Timestamp("2018-10-01")
        mock_load_config.return_value = cfg
        mock_fetch.return_value = sample_orders_df.copy()

        result, _ = build_labels()

        row = result[result["customer_unique_id"] == "cust_churned"].iloc[0]
        assert row["label"] == 1
        assert row["censored"] == False  # noqa: E712

    @patch("app.Features.build_churn_label.fetch_valid_orders")
    @patch("app.Features.build_churn_label.load_config")
    def test_recent_customer_labeled_censored(
        self, mock_load_config, mock_fetch, sample_orders_df, valid_config_yaml
    ):
        cfg = dict(valid_config_yaml)
        cfg["reference_date"] = pd.Timestamp("2018-10-01")
        mock_load_config.return_value = cfg
        mock_fetch.return_value = sample_orders_df.copy()

        result, _ = build_labels()

        row = result[result["customer_unique_id"] == "cust_censored"].iloc[0]
        assert row["censored"] == True  # noqa: E712
        assert pd.isna(row["label"])

    @patch("app.Features.build_churn_label.fetch_valid_orders")
    @patch("app.Features.build_churn_label.load_config")
    def test_exclude_censored_removes_censored_rows(
        self, mock_load_config, mock_fetch, sample_orders_df, valid_config_yaml
    ):
        cfg = dict(valid_config_yaml)
        cfg["reference_date"] = pd.Timestamp("2018-10-01")
        cfg["exclude_censored_customers"] = True
        mock_load_config.return_value = cfg
        mock_fetch.return_value = sample_orders_df.copy()

        result, _ = build_labels()

        assert "cust_censored" not in result["customer_unique_id"].values
        assert result["censored"].sum() == 0

    @patch("app.Features.build_churn_label.fetch_valid_orders")
    @patch("app.Features.build_churn_label.load_config")
    def test_keeps_censored_when_flag_is_false(
        self, mock_load_config, mock_fetch, sample_orders_df, valid_config_yaml
    ):
        cfg = dict(valid_config_yaml)
        cfg["reference_date"] = pd.Timestamp("2018-10-01")
        cfg["exclude_censored_customers"] = False
        mock_load_config.return_value = cfg
        mock_fetch.return_value = sample_orders_df.copy()

        result, _ = build_labels()

        assert "cust_censored" in result["customer_unique_id"].values

    @patch("app.Features.build_churn_label.fetch_valid_orders")
    @patch("app.Features.build_churn_label.load_config")
    def test_empty_orders_returns_empty_frame_with_expected_columns(
        self, mock_load_config, mock_fetch, valid_config_yaml
    ):
        cfg = dict(valid_config_yaml)
        cfg["reference_date"] = pd.Timestamp("2018-10-01")
        mock_load_config.return_value = cfg
        mock_fetch.return_value = pd.DataFrame(
            columns=[
                "customer_unique_id",
                "order_id",
                "order_status",
                "order_purchase_timestamp",
            ]
        )

        result, reference_date = build_labels()

        assert result.empty
        expected_columns = {
            "customer_unique_id",
            "first_order_date",
            "last_order_date",
            "days_since_last_order",
            "num_valid_orders",
            "censored",
            "label",
        }
        assert expected_columns.issubset(set(result.columns))
        assert reference_date == pd.Timestamp("2018-10-01")

    @patch("app.Features.build_churn_label.fetch_valid_orders")
    @patch("app.Features.build_churn_label.load_config")
    def test_output_schema_and_column_order(
        self, mock_load_config, mock_fetch, sample_orders_df, valid_config_yaml
    ):
        cfg = dict(valid_config_yaml)
        cfg["reference_date"] = pd.Timestamp("2018-10-01")
        mock_load_config.return_value = cfg
        mock_fetch.return_value = sample_orders_df.copy()

        result, _ = build_labels()

        assert list(result.columns) == [
            "customer_unique_id",
            "first_order_date",
            "last_order_date",
            "days_since_last_order",
            "num_valid_orders",
            "censored",
            "label",
        ]
        # label must be a pandas nullable integer so NA (censored) is representable
        assert str(result["label"].dtype) == "Int64"

    @patch("app.Features.build_churn_label.fetch_valid_orders")
    @patch("app.Features.build_churn_label.load_config")
    def test_boundary_gap_equal_to_window_counts_as_repeat(
        self, mock_load_config, mock_fetch, valid_config_yaml
    ):
        """A gap of EXACTLY return_window_days should count as retained,
        since the code uses <= return_window_days."""
        cfg = dict(valid_config_yaml)
        cfg["reference_date"] = pd.Timestamp("2019-01-01")
        cfg["return_window_days"] = 180
        mock_load_config.return_value = cfg

        orders = pd.DataFrame(
            {
                "customer_unique_id": ["cust_boundary", "cust_boundary"],
                "order_id": ["o1", "o2"],
                "order_status": ["delivered", "delivered"],
                "order_purchase_timestamp": pd.to_datetime(
                    ["2018-01-01", "2018-06-30"]  # exactly 180 days apart
                ),
            }
        )
        mock_fetch.return_value = orders

        result, _ = build_labels()
        row = result[result["customer_unique_id"] == "cust_boundary"].iloc[0]
        assert row["label"] == 0

    @patch("app.Features.build_churn_label.fetch_valid_orders")
    @patch("app.Features.build_churn_label.load_config")
    def test_gap_one_day_over_window_does_not_count_as_repeat(
        self, mock_load_config, mock_fetch, valid_config_yaml
    ):
        """A gap of return_window_days + 1 should NOT count as a qualifying
        repeat purchase (see build_churn_label's <= check)."""
        cfg = dict(valid_config_yaml)
        cfg["reference_date"] = pd.Timestamp("2019-01-01")
        cfg["return_window_days"] = 180
        mock_load_config.return_value = cfg

        orders = pd.DataFrame(
            {
                "customer_unique_id": ["cust_over", "cust_over"],
                "order_id": ["o1", "o2"],
                "order_status": ["delivered", "delivered"],
                "order_purchase_timestamp": pd.to_datetime(
                    ["2018-01-01", "2018-07-01"]  # 181 days apart
                ),
            }
        )
        mock_fetch.return_value = orders

        result, _ = build_labels()
        row = result[result["customer_unique_id"] == "cust_over"].iloc[0]
        # Not a qualifying repeat -> either churned or censored, never retained
        assert row["label"] != 0


# ---------------------------------------------------------------------------
# save_labels_to_db
# ---------------------------------------------------------------------------

class TestSaveLabelsToDb:

    def test_save_labels_calls_to_sql_with_expected_args(self):
        df = pd.DataFrame(
            {
                "customer_unique_id": ["c1", "c2"],
                "label": pd.array([0, 1], dtype="Int64"),
            }
        )

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        with patch.object(pd.DataFrame, "to_sql") as mock_to_sql:
            save_labels_to_db(mock_engine, df, table_name="customer_churn_labels")

            mock_to_sql.assert_called_once()
            _, kwargs = mock_to_sql.call_args
            assert kwargs["if_exists"] == "replace"
            assert kwargs["index"] is False
            assert kwargs["chunksize"] == 5000

    def test_save_labels_converts_na_label_to_none(self):
        df = pd.DataFrame(
            {
                "customer_unique_id": ["c1", "c2"],
                "label": pd.array([0, pd.NA], dtype="Int64"),
            }
        )

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        captured = {}

        def fake_to_sql(self, *args, **kwargs):
            captured["frame"] = self.copy()

        with patch.object(pd.DataFrame, "to_sql", new=fake_to_sql):
            save_labels_to_db(mock_engine, df, table_name="customer_churn_labels")

        assert captured["frame"]["label"].iloc[1] is None
        assert captured["frame"]["label"].iloc[0] == 0

    def test_save_labels_uses_default_table_name(self):
        df = pd.DataFrame(
            {
                "customer_unique_id": ["c1"],
                "label": pd.array([0], dtype="Int64"),
            }
        )
        mock_engine = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = MagicMock()

        with patch.object(pd.DataFrame, "to_sql") as mock_to_sql:
            save_labels_to_db(mock_engine, df)  # no table_name passed

            args, _ = mock_to_sql.call_args
            assert args[0] == "customer_churn_labels"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
