import numpy as np
import pandas as pd
import pytest

from app.segmentation.customer_intelligence.gmm.model_selection import (
    _composite_selection_score,
    evaluate_k_range,
    select_best_k,
)


def make_cluster_data():
    rng = np.random.default_rng(42)

    cluster_1 = rng.normal(
        loc=0,
        scale=0.3,
        size=(30, 3),
    )

    cluster_2 = rng.normal(
        loc=5,
        scale=0.3,
        size=(30, 3),
    )

    cluster_3 = rng.normal(
        loc=10,
        scale=0.3,
        size=(30, 3),
    )

    data = np.vstack(
        [
            cluster_1,
            cluster_2,
            cluster_3,
        ]
    )

    return pd.DataFrame(
        data,
        columns=[
            "feature_1",
            "feature_2",
            "feature_3",
        ],
    )


def test_evaluate_k_range_returns_expected_columns():
    data = make_cluster_data()

    result = evaluate_k_range(
        data,
        k_range=range(2, 4),
        silhouette_sample_size=50,
    )

    expected_columns = {
        "n_clusters",
        "bic",
        "silhouette",
        "davies_bouldin",
        "min_cluster_share",
        "selection_score",
    }

    assert expected_columns.issubset(
        result.columns
    )


def test_evaluate_k_range_returns_one_row_per_k():
    data = make_cluster_data()

    result = evaluate_k_range(
        data,
        k_range=range(2, 5),
        silhouette_sample_size=50,
    )

    assert len(result) == 3

    assert result["n_clusters"].tolist() == [
        2,
        3,
        4,
    ]


def test_evaluate_k_range_has_valid_cluster_shares():
    data = make_cluster_data()

    result = evaluate_k_range(
        data,
        k_range=range(2, 4),
        silhouette_sample_size=50,
    )

    assert (
        result["min_cluster_share"] > 0
    ).all()

    assert (
        result["min_cluster_share"] <= 1
    ).all()


def test_evaluate_k_range_has_selection_scores():
    data = make_cluster_data()

    result = evaluate_k_range(
        data,
        k_range=range(2, 4),
        silhouette_sample_size=50,
    )

    assert result["selection_score"].notna().all()


def test_composite_selection_score():
    df = pd.DataFrame(
        {
            "bic": [
                100.0,
                200.0,
                300.0,
            ],
            "silhouette": [
                0.8,
                0.5,
                0.2,
            ],
            "davies_bouldin": [
                0.2,
                0.5,
                0.8,
            ],
        }
    )

    result = _composite_selection_score(df)

    assert len(result) == 3
    assert result.iloc[0] < result.iloc[2]


def test_select_best_k_returns_integer():
    evaluation_df = pd.DataFrame(
        {
            "n_clusters": [2, 3, 4],
            "bic": [300, 200, 100],
            "silhouette": [0.4, 0.7, 0.5],
            "davies_bouldin": [0.8, 0.3, 0.6],
            "min_cluster_share": [0.20, 0.30, 0.25],
            "selection_score": [0.8, 0.1, 0.5],
        }
    )

    result = select_best_k(
        evaluation_df
    )

    assert isinstance(result, int)
    assert result == 3


def test_select_best_k_respects_min_cluster_share():
    evaluation_df = pd.DataFrame(
        {
            "n_clusters": [2, 3, 4],
            "bic": [100, 90, 80],
            "silhouette": [0.8, 0.7, 0.6],
            "davies_bouldin": [0.2, 0.3, 0.4],
            "min_cluster_share": [
                0.005,
                0.20,
                0.30,
            ],
            "selection_score": [
                0.01,
                0.20,
                0.30,
            ],
        }
    )

    result = select_best_k(
        evaluation_df,
        min_cluster_share=0.01,
    )

    assert result == 3


def test_select_best_k_empty_dataframe():
    evaluation_df = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="evaluation_df is empty",
    ):
        select_best_k(evaluation_df)


def test_select_best_k_falls_back_when_no_candidate_meets_floor(
    capsys,
):
    evaluation_df = pd.DataFrame(
        {
            "n_clusters": [2, 3],
            "bic": [100, 90],
            "silhouette": [0.7, 0.8],
            "davies_bouldin": [0.3, 0.2],
            "min_cluster_share": [
                0.005,
                0.008,
            ],
            "selection_score": [
                0.2,
                0.1,
            ],
        }
    )

    result = select_best_k(
        evaluation_df,
        min_cluster_share=0.01,
    )

    assert result == 3

    captured = capsys.readouterr()

    assert "No candidate k" in captured.out