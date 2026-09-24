import numpy as np
import pandas as pd

from app.segmentation.customer_intelligence.gmm.segmenter import (
    CLUSTER_PROBABILITY_COLUMN,
    SEGMENT_ID_COLUMN,
    assign_clusters,
    train_final_gmm,
)


def make_cluster_data():
    rng = np.random.default_rng(42)

    cluster_1 = rng.normal(
        loc=0,
        scale=0.2,
        size=(20, 4),
    )

    cluster_2 = rng.normal(
        loc=5,
        scale=0.2,
        size=(20, 4),
    )

    cluster_3 = rng.normal(
        loc=10,
        scale=0.2,
        size=(20, 4),
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
            "feature_4",
        ],
    )


def test_train_final_gmm_returns_model():
    data = make_cluster_data()

    model = train_final_gmm(
        data,
        n_components=3,
    )

    assert model.n_components == 3
    assert model.random_state == 42


def test_train_final_gmm_is_fitted():
    data = make_cluster_data()

    model = train_final_gmm(
        data,
        n_components=3,
    )

    assert hasattr(model, "means_")

    assert model.means_.shape == (
        3,
        4,
    )


def test_assign_clusters_returns_expected_columns():
    data = make_cluster_data()

    model = train_final_gmm(
        data,
        n_components=3,
    )

    result = assign_clusters(
        model,
        data,
    )

    assert list(result.columns) == [
        SEGMENT_ID_COLUMN,
        CLUSTER_PROBABILITY_COLUMN,
    ]


def test_assign_clusters_returns_same_number_of_rows():
    data = make_cluster_data()

    model = train_final_gmm(
        data,
        n_components=3,
    )

    result = assign_clusters(
        model,
        data,
    )

    assert len(result) == len(data)


def test_assign_clusters_segment_ids_are_valid():
    data = make_cluster_data()

    model = train_final_gmm(
        data,
        n_components=3,
    )

    result = assign_clusters(
        model,
        data,
    )

    assert (
        result["segment_id"].min()
        >= 0
    )

    assert (
        result["segment_id"].max()
        < 3
    )


def test_cluster_probability_is_between_zero_and_one():
    data = make_cluster_data()

    model = train_final_gmm(
        data,
        n_components=3,
    )

    result = assign_clusters(
        model,
        data,
    )

    assert (
        result["cluster_probability"] >= 0
    ).all()

    assert (
        result["cluster_probability"] <= 1
    ).all()


def test_assign_clusters_preserves_input_index():
    data = make_cluster_data()

    data.index = range(100, 160)

    model = train_final_gmm(
        data,
        n_components=3,
    )

    result = assign_clusters(
        model,
        data,
    )

    assert result.index.equals(
        data.index
    )