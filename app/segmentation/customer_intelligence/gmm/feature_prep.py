from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.segmentation.customer_intelligence.data_access.customer_intelligence_repository import ID_COLUMN
from app.segmentation.customer_intelligence.risk_tier.classifier import RISK_TIER_COLUMN, load_risk_tier_config

BEHAVIOURAL_FEATURES = [
    "monetary_value",
    "frequency",
    "recency_days",
    "avg_review_score",
    "avg_delivery_days",
    "avg_delivery_delay_days",
    "avg_payment_installments",
    "freight_ratio",
    "has_bad_review",
    "has_review_comment",
    "is_delayed_delivery",
]

CHURN_PROBABILITY_COLUMN = "churn_probability"


SEGMENTATION_FEATURES = BEHAVIOURAL_FEATURES + [CHURN_PROBABILITY_COLUMN]


CHURN_PROBABILITY_WEIGHT = 0.40
RISK_TIER_WEIGHT = 0.10


def _validate_feature_columns_present(
    customer_intelligence_df: pd.DataFrame, feature_columns: list[str]
) -> None:

    missing = [c for c in feature_columns if c not in customer_intelligence_df.columns]
    if missing:
        raise KeyError(
            f"SEGMENTATION_FEATURES expects column(s) {missing} that aren't in "
            "the merged customer_intelligence dataset. This means "
            "SEGMENTATION_FEATURES (gmm/feature_prep.py) doesn't match the "
            "real features_encoded schema and needs updating.\n"
            f"Columns actually available: {sorted(customer_intelligence_df.columns.tolist())}"
        )


def select_segmentation_features(
    customer_intelligence_df: pd.DataFrame,
    risk_tiers_df: pd.DataFrame,
    feature_columns: list[str] = SEGMENTATION_FEATURES,
    risk_tier_column: str = RISK_TIER_COLUMN,
) -> pd.DataFrame:

    _validate_feature_columns_present(customer_intelligence_df, feature_columns)
    working = customer_intelligence_df[[ID_COLUMN] + feature_columns].merge(
        risk_tiers_df[[ID_COLUMN, risk_tier_column]],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    required_columns = feature_columns + [risk_tier_column]
    null_mask = working[required_columns].isnull().any(axis=1)
    if null_mask.any():
        null_counts = working.loc[null_mask, required_columns].isnull().sum()
        null_counts = null_counts[null_counts > 0]
        print(
            f"Excluding {null_mask.sum()} customer(s) from GMM segmentation "
            f"with missing segmentation feature(s): {null_counts.to_dict()}"
        )
    return working.loc[~null_mask].reset_index(drop=True)


def scale_features(
    scoreable_df: pd.DataFrame,
    feature_columns: list[str] = SEGMENTATION_FEATURES,
    churn_probability_column: str = CHURN_PROBABILITY_COLUMN,
    churn_probability_weight: float = CHURN_PROBABILITY_WEIGHT,
    risk_tier_column: str = RISK_TIER_COLUMN,
    risk_tier_weight: float = RISK_TIER_WEIGHT,
) -> tuple[pd.DataFrame, StandardScaler]:

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(scoreable_df[feature_columns])
    scaled_df = pd.DataFrame(
        scaled_values, columns=feature_columns, index=scoreable_df.index
    )
    scaled_df[churn_probability_column] = (
        scaled_df[churn_probability_column] * churn_probability_weight
    )

    all_tiers = [name for _, name in load_risk_tier_config()]
    risk_tier_dummies = pd.get_dummies(
        pd.Categorical(scoreable_df[risk_tier_column], categories=all_tiers)
    ).astype(float)
    risk_tier_dummies.columns = [
        f"{risk_tier_column}__{tier}" for tier in risk_tier_dummies.columns
    ]
    risk_tier_dummies.index = scoreable_df.index
    risk_tier_dummies = risk_tier_dummies * risk_tier_weight

    return pd.concat([scaled_df, risk_tier_dummies], axis=1), scaler
