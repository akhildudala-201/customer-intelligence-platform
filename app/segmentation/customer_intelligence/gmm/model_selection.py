from __future__ import annotations

import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

RANDOM_STATE = 42  
MIN_CLUSTER_SHARE = 0.01  


STABILITY_TOLERANCE = 0.05


PINNED_K: int | None = None


SILHOUETTE_SAMPLE_SIZE = 10_000


def evaluate_k_range(
    scaled_features: pd.DataFrame,
    k_range: range = range(2, 8),
    random_state: int = RANDOM_STATE,
    silhouette_sample_size: int = SILHOUETTE_SAMPLE_SIZE,
) -> pd.DataFrame:

    rows = []
    for k in k_range:

        model = GaussianMixture(n_components=k, random_state=random_state, n_init=5)
        labels = model.fit_predict(scaled_features)
        cluster_shares = pd.Series(labels).value_counts(normalize=True)
        sample_size = min(silhouette_sample_size, len(scaled_features))
        rows.append(
            {
                "n_clusters": k,
                "bic": model.bic(scaled_features),
                "silhouette": silhouette_score(
                    scaled_features,
                    labels,
                    sample_size=sample_size,
                    random_state=random_state,
                ),
                "davies_bouldin": davies_bouldin_score(scaled_features, labels),
                "min_cluster_share": cluster_shares.min(),
            }
        )
    evaluation_df = pd.DataFrame(rows)
    evaluation_df["selection_score"] = _composite_selection_score(evaluation_df)
    return evaluation_df


def _composite_selection_score(evaluation_df: pd.DataFrame) -> pd.Series:


    def normalize(series: pd.Series, higher_is_better: bool) -> pd.Series:
        span = series.max() - series.min()
        if span == 0:
            return pd.Series(0.0, index=series.index)
        normalized = (series - series.min()) / span
        return 1 - normalized if higher_is_better else normalized

    return (
        normalize(evaluation_df["bic"], higher_is_better=False)
        + normalize(evaluation_df["silhouette"], higher_is_better=True)
        + normalize(evaluation_df["davies_bouldin"], higher_is_better=False)
    ) / 3


def select_best_k(
    evaluation_df: pd.DataFrame,
    min_cluster_share: float = MIN_CLUSTER_SHARE,
    stability_tolerance: float = STABILITY_TOLERANCE,
    pinned_k: int | None = PINNED_K,
) -> int:

    if evaluation_df.empty:
        raise ValueError("evaluation_df is empty — nothing to select from.")

    viable = evaluation_df[evaluation_df["min_cluster_share"] >= min_cluster_share]
    if viable.empty:
        print(
            f"No candidate k has every cluster >= {min_cluster_share:.0%} of "
            "customers; falling back to the unconstrained best selection_score. "
            "Consider widening k_range or lowering min_cluster_share."
        )
        viable = evaluation_df

    viable = viable.sort_values("n_clusters")
    best_score_so_far = None
    data_driven_k = None
    for _, row in viable.iterrows():
        if best_score_so_far is None or row["selection_score"] < best_score_so_far - stability_tolerance:
            best_score_so_far = row["selection_score"]
            data_driven_k = int(row["n_clusters"])

    if pinned_k is None:
        return data_driven_k

    if pinned_k not in evaluation_df["n_clusters"].values:
        raise ValueError(
            f"PINNED_K={pinned_k} is outside the evaluated k_range "
            f"({sorted(evaluation_df['n_clusters'].tolist())}). Widen k_range "
            "or update PINNED_K."
        )
    pinned_row = evaluation_df.loc[evaluation_df["n_clusters"] == pinned_k].iloc[0]
    if pinned_row["min_cluster_share"] < min_cluster_share:
        print(
            f"WARNING: PINNED_K={pinned_k}'s smallest cluster is only "
            f"{pinned_row['min_cluster_share']:.1%} of customers (floor is "
            f"{min_cluster_share:.0%}) on THIS run's data. Segments may be "
            "too small to be useful this run — worth a human look.",
        )
    if data_driven_k != pinned_k:
        print(
            f"NOTE: this run's data-driven search would have picked k="
            f"{data_driven_k}, but PINNED_K={pinned_k} is in use. If this "
            "keeps happening, the data may have drifted enough that "
            "PINNED_K is worth revisiting.",
        )
    else:
        print(f"PINNED_K={pinned_k} matches this run's data-driven pick — no drift.")

    return pinned_k
