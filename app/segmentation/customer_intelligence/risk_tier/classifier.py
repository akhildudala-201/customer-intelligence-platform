from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd
import yaml

from app.segmentation.customer_intelligence.data_access.customer_intelligence_repository import ID_COLUMN

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "risk_tier_thresholds.yaml"

CHURN_PROBABILITY_COLUMN = "churn_probability"
RISK_TIER_COLUMN = "risk_tier"


@functools.lru_cache(maxsize=1)
def load_risk_tier_config(config_path: Path = CONFIG_PATH) -> list[tuple[float, str]]:

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    tiers = raw.get("tiers") or []
    if not tiers:
        raise ValueError(f"No tiers defined in {config_path}")


    parsed = [(float(t["upper_bound"]), str(t["name"])) for t in tiers]
    _validate_tier_coverage(parsed, config_path)

    parsed.sort(key=lambda pair: pair[0])
    return parsed


def _validate_tier_coverage(tiers: list[tuple[float, str]], config_path: Path) -> None:

    lower = 0.0
    for upper_bound, name in tiers:
        if upper_bound <= lower:
            raise ValueError(
                f"{config_path}: tier '{name}' upper_bound ({upper_bound}) "
                f"must be greater than the previous tier's upper_bound ({lower})."
            )
        lower = upper_bound

    if abs(tiers[-1][0] - 1.0) > 1e-9:
        raise ValueError(
            f"{config_path}: highest tier upper_bound is {tiers[-1][0]}, "
            "expected 1.0 to cover the full probability range."
        )


def classify_risk_tier(churn_probability: float) -> str:
    """Map a single churn probability to its configured risk tier name."""
    for upper_bound, name in load_risk_tier_config():
        if churn_probability <= upper_bound:
            return name

    raise ValueError(f"churn_probability {churn_probability} has no matching tier.")


def classify_risk_tiers(customer_intelligence_df: pd.DataFrame) -> pd.DataFrame:

    result = customer_intelligence_df[[ID_COLUMN, CHURN_PROBABILITY_COLUMN]].copy()
    result[RISK_TIER_COLUMN] = result[CHURN_PROBABILITY_COLUMN].apply(classify_risk_tier)
    return result
