from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from app.segmentation.customer_intelligence.gmm.model_selection import RANDOM_STATE

SEGMENT_ID_COLUMN = "segment_id"
CLUSTER_PROBABILITY_COLUMN = "cluster_probability"


def train_final_gmm(
    scaled_features: pd.DataFrame,
    n_components: int,
    random_state: int = RANDOM_STATE,
    n_init: int = 10,
) -> GaussianMixture:
    model = GaussianMixture(
        n_components=n_components, random_state=random_state, n_init=n_init
    )
    model.fit(scaled_features)
    return model


def assign_clusters(model: GaussianMixture, scaled_features: pd.DataFrame) -> pd.DataFrame:

    membership_probabilities = model.predict_proba(scaled_features)
    return pd.DataFrame(
        {
            SEGMENT_ID_COLUMN: model.predict(scaled_features),
            CLUSTER_PROBABILITY_COLUMN: np.max(membership_probabilities, axis=1),
        },
        index=scaled_features.index,
    )
