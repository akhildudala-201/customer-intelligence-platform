import runpy
import sys
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pandas.errors import MergeError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "app" / "Features"
for path in (str(PROJECT_ROOT), str(FEATURES_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
# pyrefly: ignore [missing-import]
try:
    import app.Features.merge as merge
except ModuleNotFoundError:
    import merge


@pytest.fixture
def reference_date() -> pd.Timestamp:
    return pd.Timestamp("2018-01-01 00:00:00")


@pytest.fixture
def customer_ids() -> list[str]:
    return ["cust_001", "cust_002", "cust_003"]


@pytest.fixture
def sample_labels(customer_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_unique_id": customer_ids,
        "label": [0, 1, 0],
        "censored": [False, False, False],
        "unrelated_metadata": ["meta_1", "meta_2", "meta_3"],
    })


@pytest.fixture
def sample_customer_base(customer_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"customer_unique_id": customer_ids})


@pytest.fixture
def sample_feature_tables(customer_ids: list[str]) -> Dict[str, pd.DataFrame]:
    return {
        "rfm": pd.DataFrame({"customer_unique_id": customer_ids, "recency_days": [15, 45, 90]}),
        "order_behavior": pd.DataFrame({"customer_unique_id": customer_ids, "order_count": [1, 3, 1]}),
        "payment": pd.DataFrame({"customer_unique_id": customer_ids, "preferred_payment_type": ["credit", "boleto", "voucher"]}),
        "reviews": pd.DataFrame({"customer_unique_id": customer_ids, "avg_review_score": [5.0, 3.2, 4.0]}),
        "products": pd.DataFrame({"customer_unique_id": customer_ids, "unique_product_count": [1, 5, 1]}),
        "fulfillment": pd.DataFrame({"customer_unique_id": customer_ids, "late_delivery_count": [0, 1, 0]}),
        "time_behavior": pd.DataFrame({"customer_unique_id": customer_ids, "tenure_days": [100, 250, 10]}),
        "geography": pd.DataFrame({"customer_unique_id": customer_ids, "customer_state": ["SP", "RJ", "MG"]}),
    }


@pytest.fixture
def sample_merged_features(
    sample_customer_base: pd.DataFrame,
    sample_feature_tables: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    df = sample_customer_base.copy()
    for table in sample_feature_tables.values():
        df = df.merge(table, on="customer_unique_id", how="left")
    return df


@pytest.fixture
def mock_pipeline():
    with (
        patch.object(merge, "build_labels") as m_labels,
        patch.object(merge, "build_customer_base") as m_base,
        patch.object(merge, "build_rfm_features") as m_rfm,
        patch.object(merge, "build_order_features") as m_order,
        patch.object(merge, "build_payment_features") as m_pay,
        patch.object(merge, "build_review_features") as m_rev,
        patch.object(merge, "build_product_features") as m_prod,
        patch.object(merge, "build_fulfillment_features") as m_ful,
        patch.object(merge, "build_time_features") as m_time,
        patch.object(merge, "build_geography_features") as m_geo,
        patch.object(merge, "merge_features") as m_merge,
        patch.object(merge, "clean_features", side_effect=lambda df: df) as m_clean,
        patch.object(merge, "validate_features", side_effect=lambda df, base: df) as m_val,
    ):
        yield {
            "build_labels": m_labels,
            "build_customer_base": m_base,
            "build_rfm_features": m_rfm,
            "build_order_features": m_order,
            "build_payment_features": m_pay,
            "build_review_features": m_rev,
            "build_product_features": m_prod,
            "build_fulfillment_features": m_ful,
            "build_time_features": m_time,
            "build_geography_features": m_geo,
            "merge_features": m_merge,
            "clean_features": m_clean,
            "validate_features": m_val,
        }


class TestMergeModuleConstants:
    def test_table_name_is_correct(self):
        assert merge.TABLE_NAME == "customer_features_with_labels"


class TestBuildCombinedDatasetHappyPath:
    def test_build_combined_dataset_standard_flow(
        self,
        mock_pipeline,
        sample_labels: pd.DataFrame,
        reference_date: pd.Timestamp,
        sample_customer_base: pd.DataFrame,
        sample_feature_tables: Dict[str, pd.DataFrame],
        sample_merged_features: pd.DataFrame,
    ):
        mock_pipeline["build_labels"].return_value = (sample_labels.copy(), reference_date)
        mock_pipeline["build_customer_base"].return_value = sample_customer_base.copy()

        mock_pipeline["build_rfm_features"].return_value = sample_feature_tables["rfm"]
        mock_pipeline["build_order_features"].return_value = sample_feature_tables["order_behavior"]
        mock_pipeline["build_payment_features"].return_value = sample_feature_tables["payment"]
        mock_pipeline["build_review_features"].return_value = sample_feature_tables["reviews"]
        mock_pipeline["build_product_features"].return_value = sample_feature_tables["products"]
        mock_pipeline["build_fulfillment_features"].return_value = sample_feature_tables["fulfillment"]
        mock_pipeline["build_time_features"].return_value = sample_feature_tables["time_behavior"]
        mock_pipeline["build_geography_features"].return_value = sample_feature_tables["geography"]

        mock_pipeline["merge_features"].return_value = sample_merged_features.copy()

        result = merge.build_combined_dataset()

        mock_pipeline["build_labels"].assert_called_once()
        mock_pipeline["build_customer_base"].assert_called_once_with(reference_date)

        for key in [
            "build_rfm_features", "build_order_features", "build_payment_features",
            "build_review_features", "build_product_features", "build_fulfillment_features",
            "build_time_features", "build_geography_features",
        ]:
            mock_pipeline[key].assert_called_once_with(reference_date)

        mock_pipeline["merge_features"].assert_called_once()
        mock_pipeline["clean_features"].assert_called_once()
        mock_pipeline["validate_features"].assert_called_once()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_labels)
        assert "churn_label" in result.columns
        assert "label" not in result.columns
        assert "censored" in result.columns
        assert "reference_date" in result.columns
        assert (result["reference_date"] == reference_date).all()
        assert result["churn_label"].tolist() == [0, 1, 0]
        assert result["customer_unique_id"].tolist() == ["cust_001", "cust_002", "cust_003"]

    def test_build_combined_dataset_drops_extraneous_label_columns(
        self,
        mock_pipeline,
        reference_date: pd.Timestamp,
    ):
        labels_with_extra = pd.DataFrame({
            "customer_unique_id": ["cust_001"],
            "label": [1],
            "censored": [False],
            "extra_col_internal_id": [999],
        })
        mock_pipeline["build_labels"].return_value = (labels_with_extra, reference_date)
        mock_pipeline["build_customer_base"].return_value = pd.DataFrame({"customer_unique_id": ["cust_001"]})

        features = pd.DataFrame({"customer_unique_id": ["cust_001"], "feature_a": [42]})
        mock_pipeline["merge_features"].return_value = features.copy()

        result = merge.build_combined_dataset()

        assert "extra_col_internal_id" not in result.columns
        assert "feature_a" in result.columns
        assert "churn_label" in result.columns

    def test_build_combined_dataset_single_customer(
        self,
        mock_pipeline,
        reference_date: pd.Timestamp,
    ):
        labels_single = pd.DataFrame({
            "customer_unique_id": ["single_cust"],
            "label": [0],
            "censored": [False],
        })
        mock_pipeline["build_labels"].return_value = (labels_single, reference_date)
        mock_pipeline["build_customer_base"].return_value = pd.DataFrame({"customer_unique_id": ["single_cust"]})

        features_single = pd.DataFrame({"customer_unique_id": ["single_cust"], "tenure": [10]})
        mock_pipeline["merge_features"].return_value = features_single.copy()

        result = merge.build_combined_dataset()

        assert len(result) == 1
        assert result.iloc[0]["customer_unique_id"] == "single_cust"
        assert result.iloc[0]["churn_label"] == 0


class TestBuildCombinedDatasetEdgeCases:
    def test_build_combined_dataset_empty_dataframes(
        self,
        mock_pipeline,
        reference_date: pd.Timestamp,
    ):
        empty_labels = pd.DataFrame({
            "customer_unique_id": pd.Series(dtype="str"),
            "label": pd.Series(dtype="int64"),
            "censored": pd.Series(dtype="bool"),
        })
        mock_pipeline["build_labels"].return_value = (empty_labels, reference_date)
        mock_pipeline["build_customer_base"].return_value = pd.DataFrame({"customer_unique_id": pd.Series(dtype="str")})

        empty_features = pd.DataFrame({"customer_unique_id": pd.Series(dtype="str")})
        mock_pipeline["merge_features"].return_value = empty_features.copy()

        result = merge.build_combined_dataset()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert "churn_label" in result.columns
        assert "censored" in result.columns
        assert "reference_date" in result.columns

    @pytest.mark.parametrize(
        "label_values, expected_churn_labels",
        [
            ([0, 1, 0], [0, 1, 0]),
            ([1, 1, 1], [1, 1, 1]),
            ([0, 0, 0], [0, 0, 0]),
            ([True, False, True], [True, False, True]),
        ],
    )
    def test_build_combined_dataset_churn_label_variations(
        self,
        mock_pipeline,
        label_values,
        expected_churn_labels,
        reference_date: pd.Timestamp,
    ):
        c_ids = ["c1", "c2", "c3"]
        labels_df = pd.DataFrame({
            "customer_unique_id": c_ids,
            "label": label_values,
            "censored": [False, False, False],
        })
        mock_pipeline["build_labels"].return_value = (labels_df, reference_date)
        mock_pipeline["build_customer_base"].return_value = pd.DataFrame({"customer_unique_id": c_ids})

        features_df = pd.DataFrame({"customer_unique_id": c_ids, "feat": [1, 2, 3]})
        mock_pipeline["merge_features"].return_value = features_df.copy()

        result = merge.build_combined_dataset()
        assert result["churn_label"].tolist() == expected_churn_labels

    @pytest.mark.parametrize(
        "censored_flags",
        [
            [False, False, False],
            [True, True, True],
            [True, False, True],
        ],
    )
    def test_build_combined_dataset_censored_flag_variations(
        self,
        mock_pipeline,
        censored_flags,
        reference_date: pd.Timestamp,
    ):
        c_ids = ["c1", "c2", "c3"]
        labels_df = pd.DataFrame({
            "customer_unique_id": c_ids,
            "label": [0, 1, 0],
            "censored": censored_flags,
        })
        mock_pipeline["build_labels"].return_value = (labels_df, reference_date)
        mock_pipeline["build_customer_base"].return_value = pd.DataFrame({"customer_unique_id": c_ids})

        features_df = pd.DataFrame({"customer_unique_id": c_ids, "feat": [1, 2, 3]})
        mock_pipeline["merge_features"].return_value = features_df.copy()

        result = merge.build_combined_dataset()
        assert result["censored"].tolist() == censored_flags


class TestBuildCombinedDatasetFailureModes:
    def test_length_mismatch_raises_value_error(
        self,
        mock_pipeline,
        reference_date: pd.Timestamp,
    ):
        labels_df = pd.DataFrame({
            "customer_unique_id": ["cust_1", "cust_2", "cust_3"],
            "label": [0, 1, 0],
            "censored": [False, False, False],
        })
        mock_pipeline["build_labels"].return_value = (labels_df, reference_date)
        mock_pipeline["build_customer_base"].return_value = pd.DataFrame({"customer_unique_id": ["cust_1", "cust_2"]})

        features_df = pd.DataFrame({"customer_unique_id": ["cust_1", "cust_2"], "score": [10, 20]})
        mock_pipeline["merge_features"].return_value = features_df.copy()

        with pytest.raises(
            ValueError,
            match=r"Some churn-label records could not be matched to a feature record\.",
        ):
            merge.build_combined_dataset()

    def test_duplicate_keys_in_labels_raises_merge_error(
        self,
        mock_pipeline,
        reference_date: pd.Timestamp,
    ):
        duplicate_labels = pd.DataFrame({
            "customer_unique_id": ["cust_1", "cust_1"],
            "label": [0, 1],
            "censored": [False, False],
        })
        mock_pipeline["build_labels"].return_value = (duplicate_labels, reference_date)
        mock_pipeline["build_customer_base"].return_value = pd.DataFrame({"customer_unique_id": ["cust_1"]})

        features_df = pd.DataFrame({"customer_unique_id": ["cust_1"], "score": [10]})
        mock_pipeline["merge_features"].return_value = features_df.copy()

        with pytest.raises(MergeError):
            merge.build_combined_dataset()

    def test_duplicate_keys_in_features_raises_merge_error(
        self,
        mock_pipeline,
        reference_date: pd.Timestamp,
    ):
        labels_df = pd.DataFrame({
            "customer_unique_id": ["cust_1"],
            "label": [0],
            "censored": [False],
        })
        mock_pipeline["build_labels"].return_value = (labels_df, reference_date)
        mock_pipeline["build_customer_base"].return_value = pd.DataFrame({"customer_unique_id": ["cust_1", "cust_1"]})

        features_df = pd.DataFrame({"customer_unique_id": ["cust_1", "cust_1"], "score": [10, 20]})
        mock_pipeline["merge_features"].return_value = features_df.copy()

        with pytest.raises(MergeError):
            merge.build_combined_dataset()

    @pytest.mark.parametrize(
        "missing_col, labels_dict",
        [
            ("customer_unique_id", {"label": [0], "censored": [False]}),
            ("label", {"customer_unique_id": ["c1"], "censored": [False]}),
            ("censored", {"customer_unique_id": ["c1"], "label": [0]}),
        ],
    )
    def test_missing_required_labels_columns_raises_key_error(
        self,
        mock_pipeline,
        missing_col: str,
        labels_dict: dict,
        reference_date: pd.Timestamp,
    ):
        invalid_labels = pd.DataFrame(labels_dict)
        mock_pipeline["build_labels"].return_value = (invalid_labels, reference_date)
        mock_pipeline["build_customer_base"].return_value = pd.DataFrame({"customer_unique_id": ["c1"]})

        features_df = pd.DataFrame({"customer_unique_id": ["c1"]})
        mock_pipeline["merge_features"].return_value = features_df.copy()

        with pytest.raises(KeyError):
            merge.build_combined_dataset()

    def test_missing_customer_unique_id_in_features_raises_key_error(
        self,
        mock_pipeline,
        reference_date: pd.Timestamp,
    ):
        valid_labels = pd.DataFrame({
            "customer_unique_id": ["c1"],
            "label": [0],
            "censored": [False],
        })
        mock_pipeline["build_labels"].return_value = (valid_labels, reference_date)
        mock_pipeline["build_customer_base"].return_value = pd.DataFrame({"customer_unique_id": ["c1"]})

        features_df = pd.DataFrame({"wrong_id": ["c1"]})
        mock_pipeline["merge_features"].return_value = features_df.copy()

        with pytest.raises(KeyError):
            merge.build_combined_dataset()

    @pytest.mark.parametrize(
        "failing_step",
        [
            "build_labels",
            "build_customer_base",
            "build_rfm_features",
            "merge_features",
            "clean_features",
            "validate_features",
        ],
    )
    def test_upstream_step_exceptions_propagate(
        self,
        mock_pipeline,
        failing_step: str,
        reference_date: pd.Timestamp,
    ):
        mock_pipeline["build_labels"].return_value = (
            pd.DataFrame({"customer_unique_id": ["c1"], "label": [0], "censored": [False]}),
            reference_date,
        )
        mock_pipeline["build_customer_base"].return_value = pd.DataFrame({"customer_unique_id": ["c1"]})
        mock_pipeline["merge_features"].return_value = pd.DataFrame({"customer_unique_id": ["c1"]})

        mock_pipeline[failing_step].side_effect = RuntimeError(f"Failure in {failing_step}")

        with pytest.raises(RuntimeError, match=f"Failure in {failing_step}"):
            merge.build_combined_dataset()


class TestSaveCombinedDataset:
    def test_save_combined_dataset_calls_to_sql_with_proper_arguments(self, capsys):
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_df.shape = (450, 32)

        with patch.object(merge, "engine", MagicMock()) as mock_engine:
            merge.save_combined_dataset(mock_df)

            mock_df.to_sql.assert_called_once_with(
                "customer_features_with_labels",
                con=mock_engine,
                if_exists="replace",
                index=False,
            )

        captured = capsys.readouterr()
        assert "Combined database table replaced: customer_features_with_labels" in captured.out
        assert "Final shape: (450, 32)" in captured.out

    def test_save_combined_dataset_propagates_database_errors(self):
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_df.to_sql.side_effect = ConnectionError("Could not connect to database server")

        with pytest.raises(ConnectionError, match="Could not connect to database server"):
            merge.save_combined_dataset(mock_df)

    def test_save_combined_dataset_empty_dataframe(self, capsys):
        empty_df = pd.DataFrame(columns=["customer_unique_id", "churn_label"])

        with patch.object(merge, "engine", MagicMock()) as mock_engine:
            with patch.object(empty_df, "to_sql") as mock_to_sql:
                merge.save_combined_dataset(empty_df)

                mock_to_sql.assert_called_once_with(
                    "customer_features_with_labels",
                    con=mock_engine,
                    if_exists="replace",
                    index=False,
                )

        captured = capsys.readouterr()
        assert "Final shape: (0, 2)" in captured.out


class TestMainAndExecution:
    @patch.object(merge, "save_combined_dataset")
    @patch.object(merge, "build_combined_dataset")
    def test_main_orchestration_flow(self, mock_build, mock_save):
        sample_df = pd.DataFrame({"customer_unique_id": ["cust_x"], "churn_label": [1]})
        mock_build.return_value = sample_df

        merge.main()

        mock_build.assert_called_once()
        mock_save.assert_called_once_with(sample_df)

    @patch.object(merge, "save_combined_dataset")
    @patch.object(merge, "build_combined_dataset")
    def test_main_aborts_save_on_build_error(self, mock_build, mock_save):
        mock_build.side_effect = ValueError("Build failed")

        with pytest.raises(ValueError, match="Build failed"):
            merge.main()

        mock_save.assert_not_called()

    def test_script_execution_entrypoint(self):
        with (
            patch("build_churn_label.build_labels") as mock_labels,
            patch("build_features.build_customer_base") as mock_base,
            patch("build_features.merge_features") as mock_merge,
            patch("build_features.clean_features", side_effect=lambda x: x),
            patch("build_features.validate_features", side_effect=lambda x, y: x),
            patch("pandas.DataFrame.to_sql"),
            patch.object(sys, "argv", ["merge.py"]),
        ):
            mock_labels.return_value = (
                pd.DataFrame({"customer_unique_id": ["c1"], "label": [0], "censored": [False]}),
                pd.Timestamp("2018-01-01"),
            )
            mock_base.return_value = pd.DataFrame({"customer_unique_id": ["c1"]})
            mock_merge.return_value = pd.DataFrame({"customer_unique_id": ["c1"]})

            for feat in ["rfm", "order", "payment", "review", "product", "fulfillment", "time", "geography"]:
                patch(f"build_features.build_{feat}_features", return_value=pd.DataFrame({"customer_unique_id": ["c1"]})).start()

            result = runpy.run_path(str(FEATURES_DIR / "merge.py"), run_name="__main__")
            assert "TABLE_NAME" in result
            assert result["TABLE_NAME"] == "customer_features_with_labels"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
