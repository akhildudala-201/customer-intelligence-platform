"""
reason_code_lookup.py

WHY THIS FILE EXISTS
---------------------
Reason codes must come from reason_codes.yaml, not be hard-coded in Python.
This module is the only place that reads that YAML file and turns SHAP
feature contributions into a list of reason code strings (e.g. ["RC01",
"RC03"]). It keeps business-text/reason-code logic separate from the pure
numeric SHAP wrapper (shap_explainer.py) and from prediction assembly
(predict.py).

WHAT IT ACCEPTS
---------------
`load_reason_codes(path)` accepts a path to reason_codes.yaml.

`select_top_reason_codes(shap_values, reason_code_config, top_n)` accepts:
    - shap_values: dict of feature_name -> SHAP value (from shap_explainer)
    - reason_code_config: the dict loaded by load_reason_codes()
    - top_n: how many top contributors to consider (default 3)

WHAT IT RETURNS
----------------
`load_reason_codes` returns a dict keyed by reason code (e.g. "RC01"),
each value a dict with "feature", "direction", "message".

`select_top_reason_codes` returns a list of reason code strings (e.g.
["RC01", "RC04"]), in descending order of |SHAP value|. A feature only
produces a reason code if there is a YAML entry matching BOTH its name and
the direction (sign) of its SHAP contribution. Features with no matching
reason code are skipped (not an error — not every feature needs to have a
reason code defined).

COVERAGE NOTE
-------------
reason_codes.yaml maps 6 of the 13 columns in the real feature contract
(feature_contract.REQUIRED_FEATURES), BOTH directions each (12 codes
total) — the ones whose business direction is unambiguous from the name
alone: avg_review_score, has_bad_review, freight_ratio, avg_delivery_days,
avg_delivery_delay_days, is_delayed_delivery. The other 7 (monetary_value,
avg_payment_installments, has_review_comment, avg_product_weight_g,
dominant_product_category_frequency, customer_city_state_frequency,
preferred_payment_type_debit_card) are intentionally left unmapped rather
than guessed — select_top_reason_codes() already skips unmapped top-SHAP
features gracefully (see its docstring below), so predictions still work,
just with fewer/no reason codes on customers whose top drivers are one of
the unmapped features (this is a real, expected outcome, not a bug — a
customer's top-3 SHAP drivers landing entirely among these 7 gives an
empty reason_codes list). Nothing in this Python file needs to change to
add more coverage — only reason_codes.yaml does, once product/business
signs off on direction + message text for the remaining 7.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_reason_codes(path: str | Path) -> dict[str, dict]:
    """Load the reason code configuration from a YAML file.

    Raises FileNotFoundError if the file does not exist, and ValueError if
    the file is empty or malformed.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Reason codes config not found at '{config_path}'.")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Reason codes config at '{config_path}' is empty.")

    for code, entry in data.items():
        for required_key in ("feature", "direction", "message"):
            if required_key not in entry:
                raise ValueError(
                    f"Reason code '{code}' is missing required key '{required_key}'."
                )
        if entry["direction"] not in ("positive", "negative"):
            raise ValueError(
                f"Reason code '{code}' has invalid direction "
                f"'{entry['direction']}' (must be 'positive' or 'negative')."
            )

    return data


def select_top_reason_codes(
    shap_values: dict[str, float],
    reason_code_config: dict[str, dict],
    top_n: int = 3,
    predicted_positive: bool | None = None,
) -> list[str]:
    """
    Select the top_n SHAP contributors by absolute magnitude, and map each
    to a reason code (if one exists for that feature + direction).

    The SHAP sign is preserved for matching: a positive SHAP value only
    matches a reason code configured with direction "positive" (pushes
    churn probability up), and likewise for negative.

    `predicted_positive` (optional): whether the model's OVERALL
    prediction for this customer is churn (True) or retention (False).
    When given, only SHAP contributors whose OWN sign agrees with that
    overall prediction are eligible for top_n ranking -- for a
    predicted-churn customer, only risk-INCREASING (positive-SHAP)
    features are candidates; for predicted-retention, only risk-DECREASING
    (negative-SHAP) ones. Without this, the top-|SHAP| feature can be a
    risk-LOWERING factor even when the overall prediction is high-risk
    (a large positive driver outweighed several negative ones) -- and if
    that risk-lowering feature happens to be the only one of the top-N
    with a reason_codes.yaml entry, the customer's sole "reason" ends up
    contradicting their own predicted outcome (e.g. "orders arrive
    quickly, which lowers risk" shown as the reason for a 99% churn
    prediction). Passing predicted_positive prevents exactly that.
    Default None preserves the original behavior (no direction
    filtering) for any caller not yet updated to pass it.
    """
    # Sort features by absolute SHAP magnitude, descending.
    ranked_features = sorted(
        shap_values.items(), key=lambda item: abs(item[1]), reverse=True
    )

    if predicted_positive is not None:
        wants_positive_shap = bool(predicted_positive)
        ranked_features = [
            (feature, value)
            for feature, value in ranked_features
            if (value > 0) == wants_positive_shap
        ]

    top_features = ranked_features[:top_n]

    # Build a lookup: (feature, direction) -> reason code.
    lookup: dict[tuple[str, str], str] = {
        (entry["feature"], entry["direction"]): code
        for code, entry in reason_code_config.items()
    }

    reason_codes: list[str] = []
    for feature_name, value in top_features:
        direction = "positive" if value > 0 else "negative"
        code = lookup.get((feature_name, direction))
        if code is not None:
            reason_codes.append(code)

    return reason_codes