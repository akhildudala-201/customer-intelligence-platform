from __future__ import annotations

from collections import Counter

import pandas as pd

SEGMENT_ID_COLUMN = "segment_id"
SEGMENT_LABEL_COLUMN = "segment_label"


FREQUENCY_COLUMN = "frequency"
VALUE_COLUMN = "monetary_value"
REVIEW_SCORE_COLUMN = "avg_review_score"
BAD_REVIEW_COLUMN = "has_bad_review"
DELAYED_DELIVERY_COLUMN = "is_delayed_delivery"
DELIVERY_DELAY_DAYS_COLUMN = "avg_delivery_delay_days"
REVIEW_COMMENT_COLUMN = "has_review_comment"

_REQUIRED_PROFILE_COLUMNS = [
    FREQUENCY_COLUMN,
    VALUE_COLUMN,
    REVIEW_SCORE_COLUMN,
    BAD_REVIEW_COLUMN,
    DELAYED_DELIVERY_COLUMN,
    DELIVERY_DELAY_DAYS_COLUMN,
    REVIEW_COMMENT_COLUMN,
]


REPEAT_BUYER_FREQUENCY_THRESHOLD = 1.5


VALUE_Z_THRESHOLD = 0.75


def profile_clusters(
    scoreable_df: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:

    joined = scoreable_df[feature_columns].join(cluster_assignments[SEGMENT_ID_COLUMN])
    profile = joined.groupby(SEGMENT_ID_COLUMN)[feature_columns].mean()
    profile["customer_count"] = joined.groupby(SEGMENT_ID_COLUMN).size()
    return profile.reset_index()


def profile_risk_tier_mix(
    scoreable_df: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    risk_tier_column: str,
) -> pd.DataFrame:

    joined = scoreable_df[[risk_tier_column]].join(cluster_assignments[SEGMENT_ID_COLUMN])
    mix = (
        joined.groupby(SEGMENT_ID_COLUMN)[risk_tier_column]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .add_prefix("share_")
        .reset_index()
    )
    return mix


def suggest_segment_labels(cluster_profile: pd.DataFrame) -> pd.DataFrame:

    missing = [c for c in _REQUIRED_PROFILE_COLUMNS if c not in cluster_profile.columns]
    if missing:
        raise ValueError(
            f"cluster_profile is missing column(s) {missing} needed to "
            "suggest labels — check feature_columns passed to profile_clusters()."
        )

    labels = [_label_for_cluster(cluster_profile, idx) for idx in cluster_profile.index]
    labels = _dedupe(labels, cluster_profile)

    return pd.DataFrame(
        {
            SEGMENT_ID_COLUMN: cluster_profile[SEGMENT_ID_COLUMN],
            SEGMENT_LABEL_COLUMN: labels,
        }
    )


def _label_for_cluster(cluster_profile: pd.DataFrame, idx: int) -> str:
    row = cluster_profile.loc[idx]

    loyalty = (
        "Repeat Buyers"
        if row[FREQUENCY_COLUMN] >= REPEAT_BUYER_FREQUENCY_THRESHOLD
        else "One-Time Buyers"
    )
    value_prefix = _high_value_prefix(cluster_profile, idx)
    experience = _experience_qualifier(cluster_profile, idx)

    return f"{value_prefix}{experience} {loyalty}".strip()


def _high_value_prefix(cluster_profile: pd.DataFrame, idx: int) -> str:
    values = cluster_profile[VALUE_COLUMN]
    std = values.std(ddof=0)
    if std == 0:
        return ""
    z_score = (values.loc[idx] - values.mean()) / std
    return "High-Value " if z_score > VALUE_Z_THRESHOLD else ""


def _experience_qualifier(cluster_profile: pd.DataFrame, idx: int) -> str:

    row = cluster_profile.loc[idx]

    is_late_and_flagged_delayed = (
        row[DELAYED_DELIVERY_COLUMN] >= 0.5 and row[DELIVERY_DELAY_DAYS_COLUMN] > 0
    )
    if is_late_and_flagged_delayed:
        return "Delayed-Delivery Frustrated"

    if row[BAD_REVIEW_COLUMN] >= 0.5:
        return "Product-Dissatisfied"

    below_median_review = row[REVIEW_SCORE_COLUMN] < cluster_profile[REVIEW_SCORE_COLUMN].median()
    above_median_bad_reviews = row[BAD_REVIEW_COLUMN] > cluster_profile[BAD_REVIEW_COLUMN].median()
    if below_median_review and above_median_bad_reviews:
        return "Service-Frustrated"

    if row[REVIEW_COMMENT_COLUMN] >= cluster_profile[REVIEW_COMMENT_COLUMN].quantile(0.6):
        return "Review-Engaged"

    return "Satisfied"


def _dedupe(labels: list[str], cluster_profile: pd.DataFrame) -> list[str]:

    counts = Counter(labels)
    if all(count == 1 for count in counts.values()):
        return labels

    out = []
    seen = Counter()
    for label, segment_id in zip(labels, cluster_profile[SEGMENT_ID_COLUMN]):
        if counts[label] > 1:
            seen[label] += 1
            print(
                f"  NOTE: segment_id={segment_id} shares the suggested label "
                f"'{label}' with another segment — the automatic rules found "
                "no distinguishing signal between them. Review profile_clusters() "
                "output for these segments and rename manually."
            )
            out.append(f"{label} ({seen[label]})")
        else:
            out.append(label)
    return out
