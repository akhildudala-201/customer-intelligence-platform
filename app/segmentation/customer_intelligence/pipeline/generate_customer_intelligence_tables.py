from __future__ import annotations

import os

_logical_cpu_count = os.cpu_count() or 2
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, _logical_cpu_count - 1)))

import pandas as pd
from sqlalchemy import text

from app.Database.database import engine
from app.segmentation.customer_intelligence.data_access.customer_intelligence_repository import (
    ID_COLUMN,
    build_customer_intelligence_base,
)
from app.segmentation.customer_intelligence.gmm.feature_prep import (
    SEGMENTATION_FEATURES,
    scale_features,
    select_segmentation_features,
)
from app.segmentation.customer_intelligence.gmm.labeling import (
    profile_clusters,
    profile_risk_tier_mix,
    suggest_segment_labels,
)
from app.segmentation.customer_intelligence.gmm.model_selection import evaluate_k_range, select_best_k
from app.segmentation.customer_intelligence.gmm.segmenter import assign_clusters, train_final_gmm
from app.segmentation.customer_intelligence.risk_tier.classifier import RISK_TIER_COLUMN, classify_risk_tiers

CUSTOMER_INTELLIGENCE_BASE_TABLE = "customer_intelligence_base"
RISK_TIER_TABLE_NAME = "customer_risk_tiers"
SEGMENTS_TABLE_NAME = "customer_segments"
K_SEARCH_RANGE = range(2, 8)
SAMPLE_ROW_COUNT = 5

K = 3

_SECTION_RULE = "=" * 78
_STEP_RULE = "-" * 78


def _section(title: str) -> None:

    print(f"\n{_SECTION_RULE}\n{title}\n{_SECTION_RULE}")


def _round_for_display(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:

    display_df = df.copy()
    float_columns = display_df.select_dtypes(include="float").columns
    display_df[float_columns] = display_df[float_columns].round(decimals)
    return display_df


def _build_segment_summary(profile: pd.DataFrame, label_map: pd.DataFrame) -> pd.DataFrame:

    summary = label_map.merge(
        profile[["segment_id", "customer_count"]], on="segment_id", how="left"
    )
    summary["pct_of_customers"] = (
        summary["customer_count"] / summary["customer_count"].sum() * 100
    ).round(1)
    summary = summary.sort_values("customer_count", ascending=False).reset_index(drop=True)
    summary["customer_count"] = summary["customer_count"].map("{:,.0f}".format)
    summary["pct_of_customers"] = summary["pct_of_customers"].map("{:.1f}%".format)
    return summary


def _warn_if_case_variant_duplicate_exists(table_name: str) -> None:

    query = text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = DATABASE() "
        "AND LOWER(table_name) = LOWER(:table_name) "
        "AND table_name != :table_name"
    )
    with engine.connect() as connection:
        case_variants = [
            row[0] for row in connection.execute(query, {"table_name": table_name})
        ]
    if case_variants:
        print(
            f"  WARNING: found table name(s) differing only by case from "
            f"`{table_name}`: {case_variants}. MySQL treats these as SEPARATE "
            f"tables, and this script only dropped/rewrote the exact name "
            f"`{table_name}` — the case-variant one(s) above were NOT "
            "touched. If they're leftovers from before this pipeline "
            f"existed, drop them manually once you've confirmed they're "
            f"stale, e.g.: DROP TABLE `{case_variants[0]}`;"
        )


def _validate_before_write(df: pd.DataFrame, table_name: str, id_column: str = ID_COLUMN) -> None:

    if df.empty:
        raise ValueError(
            f"Refusing to write `{table_name}`: the DataFrame is empty. "
            "The existing table (if any) was left untouched."
        )

    if id_column in df.columns:
        duplicate_ids = df[df.duplicated(subset=[id_column], keep=False)]
        if not duplicate_ids.empty:
            raise ValueError(
                f"Refusing to write `{table_name}`: {duplicate_ids[id_column].nunique()} "
                f"{id_column} value(s) appear more than once "
                f"({duplicate_ids[id_column].value_counts().head(5).to_dict()}, "
                "showing up to 5). The existing table (if any) was left untouched."
            )


def _write_table(df: pd.DataFrame, table_name: str) -> None:

    _validate_before_write(df, table_name)
    _warn_if_case_variant_duplicate_exists(table_name)
    with engine.begin() as connection:
        connection.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)


def _print_table_sample(table_name: str, n: int = SAMPLE_ROW_COUNT) -> None:

    sample_df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT {n}", con=engine)
    print(f"\n  `{table_name}` — {n} row(s) read back from the database:")
    print(_round_for_display(sample_df).to_string(index=False))


def generate_risk_tiers(customer_intelligence_df: pd.DataFrame) -> pd.DataFrame:
    """Schema 6.1 output: customer_unique_id, churn_probability, risk_tier."""
    return classify_risk_tiers(customer_intelligence_df)


def generate_segments(
    customer_intelligence_df: pd.DataFrame, risk_tiers_df: pd.DataFrame
) -> pd.DataFrame:

    print("Step 1/6 — selecting behavioural features + risk_tier, dropping incomplete rows...")
    scoreable_df = select_segmentation_features(customer_intelligence_df, risk_tiers_df)
    print(
        f"  -> {len(scoreable_df)} customers scoreable on "
        f"{len(SEGMENTATION_FEATURES)} behavioural feature(s) + risk_tier: "
        f"{SEGMENTATION_FEATURES}"
    )

    print(_STEP_RULE)
    print("Step 2/6 — scaling features (StandardScaler + hybrid weights)...")
    scaled_df, _scaler = scale_features(scoreable_df)
    print(
        f"  -> scaled matrix shape: {scaled_df.shape} "
        "(behavioural features weight 1.0, churn_probability weight 0.40, "
        "one-hot risk_tier weight 0.10)"
    )

    print(_STEP_RULE)
    print(
        f"Step 3/6 — searching k = {K_SEARCH_RANGE.start}..{K_SEARCH_RANGE.stop - 1} "
        "with BIC + Silhouette + Davies-Bouldin + a practical minimum cluster size..."
    )
    k_evaluation = evaluate_k_range(scaled_df, k_range=K_SEARCH_RANGE)
    auto_best_k = select_best_k(k_evaluation)
    print(_round_for_display(k_evaluation).to_string(index=False))
    if K is not None:
        best_k = K
        print(f"  -> Selected k = {best_k} ")

    print(_STEP_RULE)
    print(f"Step 4/6 — training the final GMM (k={best_k}) and assigning clusters...")
    final_model = train_final_gmm(scaled_df, n_components=best_k)
    cluster_assignments = assign_clusters(final_model, scaled_df)
    print(f"  -> {len(cluster_assignments)} customers assigned a segment_id.")

    print(_STEP_RULE)
    print("Step 5/6 — profiling clusters and suggesting business labels...")
    profile = profile_clusters(scoreable_df, cluster_assignments, SEGMENTATION_FEATURES)
    risk_tier_mix = profile_risk_tier_mix(scoreable_df, cluster_assignments, RISK_TIER_COLUMN)
    label_map = suggest_segment_labels(profile)
    print("  Cluster profile (mean of each feature, per segment):")
    print(_round_for_display(profile).to_string(index=False))
    print("  Risk-tier composition per segment (risk_tier is a GMM input now — see feature_prep.py):")
    print(_round_for_display(risk_tier_mix).to_string(index=False))
    print("  Segment summary (label + size — the headline result of this step):")
    print(_build_segment_summary(profile, label_map).to_string(index=False))

    print(_STEP_RULE)
    print("Step 6/6 — assembling the final segments table...")
    segments = (
        scoreable_df[[ID_COLUMN]]
        .join(cluster_assignments)
        .merge(label_map, on="segment_id", how="left")
        .merge(risk_tiers_df[[ID_COLUMN, RISK_TIER_COLUMN]], on=ID_COLUMN, how="left")
    )
    print(f"  -> {len(segments)} row(s): {list(segments.columns)}")

    return segments


def generate_customer_intelligence_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    _section("[1/3] Building and validating customer_intelligence_base")
    customer_intelligence_df = build_customer_intelligence_base()
    print(
        f"  -> {len(customer_intelligence_df)} rows, "
        f"{len(customer_intelligence_df.columns)} columns."
    )
    _write_table(customer_intelligence_df, CUSTOMER_INTELLIGENCE_BASE_TABLE)
    _print_table_sample(CUSTOMER_INTELLIGENCE_BASE_TABLE)

    _section("[2/3] Classifying risk tiers")
    risk_tiers_df = generate_risk_tiers(customer_intelligence_df)
    print(f"  -> {len(risk_tiers_df)} rows.")
    print(risk_tiers_df[RISK_TIER_COLUMN].value_counts().to_string())
    _write_table(risk_tiers_df, RISK_TIER_TABLE_NAME)
    _print_table_sample(RISK_TIER_TABLE_NAME)

    _section("[3/3] Running GMM segmentation")
    segments_df = generate_segments(customer_intelligence_df, risk_tiers_df)
    _write_table(segments_df, SEGMENTS_TABLE_NAME)
    _print_table_sample(SEGMENTS_TABLE_NAME)

    return customer_intelligence_df, risk_tiers_df, segments_df


if __name__ == "__main__":
    base_result, risk_tiers_result, segments_result = generate_customer_intelligence_tables()
    _section("Done")
    print(
        f"Wrote {len(base_result)} row(s) to `{CUSTOMER_INTELLIGENCE_BASE_TABLE}`, "
        f"{len(risk_tiers_result)} row(s) to `{RISK_TIER_TABLE_NAME}`, "
        f"{len(segments_result)} row(s) to `{SEGMENTS_TABLE_NAME}`."
    )
