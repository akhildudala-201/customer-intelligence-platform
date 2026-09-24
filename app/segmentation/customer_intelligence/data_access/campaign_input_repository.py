from __future__ import annotations

import pandas as pd

from app.Database.database import engine

RISK_TIERS_TABLE = "customer_risk_tiers"
SEGMENTS_TABLE = "customer_segments"
CLV_TABLE = "customer_clv"

ID_COLUMN = "customer_unique_id"
RISK_TIER_COLUMN = "risk_tier"
SEGMENT_LABEL_COLUMN = "segment_label"
VALUE_TIER_COLUMN = "value_tier"
CLV_COLUMN = "clv"


def get_risk_tiers() -> pd.DataFrame:
    """customer_unique_id, risk_tier — from customer_risk_tiers (schema 6.1)."""
    df = pd.read_sql_table(RISK_TIERS_TABLE, con=engine)
    _require_columns(df, [ID_COLUMN, RISK_TIER_COLUMN], RISK_TIERS_TABLE)
    return df[[ID_COLUMN, RISK_TIER_COLUMN]].copy()


def get_segments() -> pd.DataFrame:
    """customer_unique_id, segment_label — from customer_segments (schema 6.2)."""
    df = pd.read_sql_table(SEGMENTS_TABLE, con=engine)
    _require_columns(df, [ID_COLUMN, SEGMENT_LABEL_COLUMN], SEGMENTS_TABLE)
    return df[[ID_COLUMN, SEGMENT_LABEL_COLUMN]].copy()


def get_clv() -> pd.DataFrame:

    df = pd.read_sql_table(CLV_TABLE, con=engine)
    _require_columns(df, [ID_COLUMN, VALUE_TIER_COLUMN, CLV_COLUMN], CLV_TABLE)
    return df[[ID_COLUMN, VALUE_TIER_COLUMN, CLV_COLUMN]].copy()


def build_campaign_input_base() -> pd.DataFrame:

    risk_tiers_df = get_risk_tiers()
    segments_df = get_segments()
    clv_df = get_clv()

    merged = risk_tiers_df.merge(segments_df, on=ID_COLUMN, how="inner", validate="one_to_one")
    merged = merged.merge(clv_df, on=ID_COLUMN, how="inner", validate="one_to_one")

    _report_unmatched_customers(risk_tiers_df, segments_df, clv_df, merged)
    _validate_merged_input(merged)

    return merged


def _require_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"`{table_name}` is missing expected column(s) {missing}. "
            f"Available columns: {sorted(map(str, df.columns.tolist()))}"
        )


def _report_unmatched_customers(
    risk_tiers_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    clv_df: pd.DataFrame,
    merged: pd.DataFrame,
) -> None:

    merged_ids = set(merged[ID_COLUMN])
    sources = {
        RISK_TIERS_TABLE: set(risk_tiers_df[ID_COLUMN]),
        SEGMENTS_TABLE: set(segments_df[ID_COLUMN]),
        CLV_TABLE: set(clv_df[ID_COLUMN]),
    }
    for table_name, ids in sources.items():
        only_here = ids - merged_ids
        if only_here:
            print(
                f"{len(only_here)} customer(s) present in `{table_name}` did not "
                "match across all three inputs (customer_risk_tiers, "
                "customer_segments, customer_clv) — excluded from campaign "
                "recommendations this run."
            )


def _validate_merged_input(merged: pd.DataFrame) -> None:
    if merged.empty:
        raise ValueError(
            "Merged campaign-input base is empty — no overlap across "
            f"`{RISK_TIERS_TABLE}`, `{SEGMENTS_TABLE}`, `{CLV_TABLE}`. Run the "
            "Person 1 (risk tier + segmentation) and Person 2 (CLV) pipelines "
            "first."
        )

    duplicate_ids = merged[merged.duplicated(subset=[ID_COLUMN], keep=False)]
    if not duplicate_ids.empty:
        raise ValueError(
            f"Duplicate {ID_COLUMN} values after merge "
            f"({duplicate_ids[ID_COLUMN].nunique()} ids) — expected one row per "
            "customer."
        )

    required = [RISK_TIER_COLUMN, SEGMENT_LABEL_COLUMN, VALUE_TIER_COLUMN]
    null_counts = merged[required].isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        raise ValueError(f"Merged campaign-input base has null value(s): {null_counts.to_dict()}")
