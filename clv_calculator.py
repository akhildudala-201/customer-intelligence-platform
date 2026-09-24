
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.Database.database import engine

FEATURES_TABLE = "customer_features_with_labels"

ID_COLUMN = "customer_unique_id"

FREQUENCY_COLUMN = "frequency"
AVG_ORDER_VALUE_COLUMN = "avg_order_value"
TENURE_DAYS_COLUMN = "tenure_days"
SINGLE_ORDER_COLUMN = "single_order_customer"

PURCHASE_FREQUENCY_COLUMN = "purchase_frequency_per_year"
LIFESPAN_COLUMN = "customer_lifespan_years"
CLV_COLUMN = "clv"
VALUE_TIER_COLUMN = "value_tier"
VALID_VALUE_TIERS = ("High", "Medium", "Low")

DAYS_PER_YEAR = 365.25


def get_customer_features() -> pd.DataFrame:
    df = pd.read_sql_table(FEATURES_TABLE, con=engine)

    if df.empty:
        raise ValueError(f"`{FEATURES_TABLE}` returned no rows.")

    required = {
        ID_COLUMN,
        FREQUENCY_COLUMN,
        AVG_ORDER_VALUE_COLUMN,
        TENURE_DAYS_COLUMN,
        SINGLE_ORDER_COLUMN,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"`{FEATURES_TABLE}` is missing expected column(s) {missing} — "
            "check the table hasn't been renamed or restructured."
        )

    return df


@dataclass(frozen=True)
class ValueTierThresholds:
    """CLV >= high_min -> High; CLV >= medium_min -> Medium; else Low."""

    high_min: float
    medium_min: float

    def __post_init__(self) -> None:
        if self.high_min < self.medium_min:
            raise ValueError(
                f"high_min ({self.high_min}) must be >= medium_min ({self.medium_min})."
            )

    def assign(self, clv: float) -> str:
        if clv >= self.high_min:
            return "High"
        if clv >= self.medium_min:
            return "Medium"
        return "Low"


def _default_thresholds(clv: pd.Series) -> ValueTierThresholds:
    """Percentile-based default: top third of customers by CLV -> High,
    middle third -> Medium, bottom third -> Low. Pass your own
    ValueTierThresholds to compute_customer_clv() for fixed business
    thresholds instead (e.g. "High = CLV > R$500")."""
    return ValueTierThresholds(
        high_min=clv.quantile(2 / 3),
        medium_min=clv.quantile(1 / 3),
    )


def _cohort_lifespan_years(features: pd.DataFrame) -> float:
    """Average observed tenure (years) among repeat customers — used as the
    single shared 'expected relationship length' for every customer. See
    module docstring for why this must NOT be each customer's own tenure."""
    repeat = features[features[SINGLE_ORDER_COLUMN] == 0]
    if repeat.empty:
        raise ValueError(
            "No repeat customers (single_order_customer == 0) found — "
            "cannot estimate a cohort customer_lifespan_years. Pass "
            "lifespan_years explicitly to compute_customer_clv()."
        )
    return float((repeat[TENURE_DAYS_COLUMN] / DAYS_PER_YEAR).mean())


def _cohort_annual_frequency(features: pd.DataFrame) -> float:
    """Average annualized purchase rate among repeat customers — the
    fallback for one-time buyers, who have no individual rate to compute
    (tenure_days == 0 for them, so frequency / years is undefined)."""
    repeat = features[features[SINGLE_ORDER_COLUMN] == 0]
    years = repeat[TENURE_DAYS_COLUMN] / DAYS_PER_YEAR
    return float((repeat[FREQUENCY_COLUMN] / years).mean())


def compute_customer_clv(
    features: pd.DataFrame | None = None,
    lifespan_years: float | None = None,
    value_tier_thresholds: ValueTierThresholds | None = None,
) -> pd.DataFrame:
   
    if features is None:
        features = get_customer_features()

    features = features.copy()

    if lifespan_years is None:
        lifespan_years = _cohort_lifespan_years(features)
    if lifespan_years <= 0:
        raise ValueError(f"customer_lifespan_years must be positive, got {lifespan_years}.")

    cohort_annual_frequency = _cohort_annual_frequency(features)

    years_observed = features[TENURE_DAYS_COLUMN] / DAYS_PER_YEAR
    individual_frequency = features[FREQUENCY_COLUMN] / years_observed.replace(0, np.nan)
    # One-time buyers (tenure_days == 0) get NaN above -> fall back to the
    # cohort's average annual frequency instead of an undefined rate.
    features[PURCHASE_FREQUENCY_COLUMN] = individual_frequency.fillna(cohort_annual_frequency)

    features[LIFESPAN_COLUMN] = lifespan_years

    features[CLV_COLUMN] = (
        features[AVG_ORDER_VALUE_COLUMN]
        * features[PURCHASE_FREQUENCY_COLUMN]
        * features[LIFESPAN_COLUMN]
    )

    thresholds = value_tier_thresholds or _default_thresholds(features[CLV_COLUMN])
    features[VALUE_TIER_COLUMN] = features[CLV_COLUMN].apply(thresholds.assign)

    _validate_clv_output(features)
    return features


def _validate_clv_output(features: pd.DataFrame) -> None:
    """Same philosophy as customer_intelligence_repository.validate_merged_dataset:
    raise loudly here rather than let a bad CLV silently become a wrong
    value tier and a wrong campaign downstream."""
    if features[CLV_COLUMN].isnull().any():
        raise ValueError(f"{features[CLV_COLUMN].isnull().sum()} row(s) have a null CLV.")

    if (features[CLV_COLUMN] < 0).any():
        n_negative = (features[CLV_COLUMN] < 0).sum()
        raise ValueError(
            f"{n_negative} row(s) have negative CLV — check for negative "
            f"{AVG_ORDER_VALUE_COLUMN} values in customer_features_with_labels."
        )

    bad_tiers = set(features[VALUE_TIER_COLUMN].unique()) - set(VALID_VALUE_TIERS)
    if bad_tiers:
        raise ValueError(f"Unexpected value_tier label(s): {bad_tiers}")

    duplicate_ids = features[features.duplicated(subset=[ID_COLUMN], keep=False)]
    if not duplicate_ids.empty:
        raise ValueError(
            f"Duplicate {ID_COLUMN} values in CLV output "
            f"({duplicate_ids[ID_COLUMN].nunique()} ids) — expected one row per customer."
        )
