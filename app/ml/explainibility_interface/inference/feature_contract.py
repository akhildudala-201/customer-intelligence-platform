"""
feature_contract.py

WHY THIS FILE EXISTS
---------------------
Every other module that needs to know "what features does the model expect,
in what order, with what types" imports from HERE instead of hard-coding a
list somewhere else. This is the single source of truth for the feature
contract used by the inference/explainability layer.

WHAT IT ACCEPTS
---------------
`validate_features(df)` accepts a pandas DataFrame of customer features
matching REQUIRED_FEATURES below. Inference may pass a model artifact's
persisted feature list through `required_features` when deployed model
versions use a different schema.

WHAT IT RETURNS
----------------
Returns a validated + reordered pandas DataFrame safe to pass into
`predict_proba`. Raises `FeatureValidationError` if the input does not
satisfy the contract.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# REAL feature names — the encoded/scaled output of
# app/Features/encoding_transformation.py + feature_selection_and_scaling.py
# (the model_ready_*/features_encoded tables), NOT raw customer_features
# columns. Order matters (LightGBM is order-sensitive) and must match the
# deployed model's own artifact["feature_cols"] exactly — do not reorder
# by hand.
#
# RESOLVED: app/ml/data_access/customer_feature_repository.py points
# TABLE_NAME at "features_encoded" (not raw `customer_features`), which
# carries every column listed below — confirmed against
# `DESCRIBE features_encoded`.
#
# RESOLVED: the earlier disagreement between this list and the training
# code's leakage exclusions is settled — this list was regenerated from
# check_contract_matches_model()'s own printed diff against a real
# trained LightGBM artifact (app/ml/train_lightgbm_model.py's LEAKY_COLS
# exclusions were correct):
#   - REMOVED (leaked the label, per LEAKY_COLS / NON_FEATURE_COLUMNS,
#     confirmed absent from the real artifact's feature_cols):
#     canceled_orders, shipped_orders, unavailable_orders,
#     delivered_rate, avg_items_per_order
#   - ADDED (present in the real artifact's feature_cols but never
#     listed here before — not leaky, just previously unaccounted for):
#     monetary_value
# If you retrain and see check_contract_matches_model()'s warning fire
# again, that means the trained feature set changed again (e.g. a
# feature_selection_and_scaling.py change, or LEAKY_COLS being edited) —
# re-sync this list from the warning's own "Model was trained on" line,
# don't hand-guess it.
#
# reason_codes.yaml note: RC01 (canceled_orders), RC02 (unavailable_orders)
# and RC03 (delivered_rate) map to features that are no longer part of
# this contract, so they can now never be selected by
# select_top_reason_codes() — not a crash (unmatched features are simply
# skipped), but dead entries worth pruning from reason_codes.yaml, and
# monetary_value (now a real feature) has no reason code yet.
# ---------------------------------------------------------------------------
REQUIRED_FEATURES: list[str] = [
    "monetary_value",
    "avg_payment_installments",
    "avg_review_score",
    "has_bad_review",
    "has_review_comment",
    "avg_product_weight_g",
    "freight_ratio",
    "avg_delivery_days",
    "avg_delivery_delay_days",
    "is_delayed_delivery",
    "dominant_product_category_frequency",
    "customer_city_state_frequency",
    "preferred_payment_type_debit_card",

]

FEATURE_DTYPES: dict[str, str] = {feature: "numeric" for feature in REQUIRED_FEATURES}
# ---------------------------------------------------------------------------
# END FEATURE CONTRACT
# ---------------------------------------------------------------------------

# Matches the primary key used throughout the Feature Engineering /
# Database modules (app/Database, app/Features) — NOT the raw `customer_id`
# (which is per-order); `customer_unique_id` is the de-duplicated
# customer-level identifier this whole pipeline keys on.
ID_COLUMN = "customer_unique_id"


class FeatureValidationError(ValueError):
    """Raised when input features do not satisfy the model's feature contract."""


def validate_features(
    df: pd.DataFrame,
    required_features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Validate a DataFrame of customer features against REQUIRED_FEATURES.

    Checks performed (in order):
      1. The DataFrame is not empty.
      2. The ID_COLUMN is present.
      3. No required feature columns are missing.
      4. Required feature values are not null.
      5. Required feature columns are numeric.
      6. Required feature columns are reordered to match REQUIRED_FEATURES
         exactly, since some models (including LightGBM) are sensitive to
         column order matching what they were trained on.

    Does NOT silently continue if required features are missing or null —
    raises FeatureValidationError instead. Extra columns are tolerated and
    dropped, since upstream tables may carry more columns than this model
    version needs.

    Returns a new DataFrame containing ID_COLUMN + REQUIRED_FEATURES, in
    that exact column order.
    """
    features = list(required_features or REQUIRED_FEATURES)

    if df is None or len(df) == 0:
        raise FeatureValidationError("Input feature data is empty.")

    if ID_COLUMN not in df.columns:
        raise FeatureValidationError(
            f"Missing required identifier column: '{ID_COLUMN}'."
        )

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise FeatureValidationError(
            f"Missing required feature column(s): {missing}"
        )

    null_counts = df[features].isnull().sum()
    bad_columns = null_counts[null_counts > 0]
    if not bad_columns.empty:
        raise FeatureValidationError(
            f"Required feature(s) contain null values: {bad_columns.to_dict()}"
        )

    non_numeric = [
        col
        for col in features
        if FEATURE_DTYPES.get(col, "numeric") == "numeric"
        and not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise FeatureValidationError(
            f"Feature(s) expected to be numeric but are not: {non_numeric}"
        )

    ordered_columns = [ID_COLUMN] + features
    validated = df[ordered_columns].copy()
    return validated


def get_model_feature_names(model) -> list[str] | None:
    """
    Best-effort extraction of the feature names a fitted model was trained
    on, so predict.py can cross-check REQUIRED_FEATURES against reality.

    Returns None if the model object doesn't expose this — callers should
    treat None as "can't verify, proceed with caution" not as an error.
    """
    feature_names = getattr(model, "feature_name_", None)
    if feature_names is not None:
        return list(feature_names)

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return list(feature_names)

    return None


def _extract_feature_names(model_or_artifact) -> list[str] | None:
    """
    Like get_model_feature_names(), but also understands the real training
    pipeline's artifact shape: `joblib.dump({"model": ..., "feature_cols":
    [...], ...}, path)` (see train_lightgbm_model.py's save_model()) rather
    than a bare fitted model. `feature_cols` in that dict is the exact,
    already-known column list the model was trained on, so it's checked
    first; falls back to get_model_feature_names() for a bare model.
    """
    if isinstance(model_or_artifact, dict):
        if "feature_cols" in model_or_artifact:
            return list(model_or_artifact["feature_cols"])
        model_or_artifact = model_or_artifact.get("model", model_or_artifact)

    return get_model_feature_names(model_or_artifact)


def check_contract_matches_model(model) -> None:
    """
    Compare REQUIRED_FEATURES against the loaded model's own record of what
    it was trained on (when available) and print a clear warning on
    mismatch. Deliberately does not raise: a mismatch here is a strong
    signal something upstream needs attention, but the pipeline should
    still run so tests/callers can see the mismatch rather than being
    blocked.

    `model` may be a bare fitted model OR the real training pipeline's
    dict artifact ({"model": ..., "feature_cols": ...}) — see
    _extract_feature_names().
    """
    model_features = _extract_feature_names(model)
    if model_features is None:
        return

    if list(model_features) != REQUIRED_FEATURES:
        print(
            "WARNING: feature_contract.REQUIRED_FEATURES does not match "
            "the feature names the loaded model was trained on.\n"
            f"  Contract expects : {REQUIRED_FEATURES}\n"
            f"  Model was trained on: {model_features}\n"
            "Update app/ml/inference/feature_contract.py to match the model's "
            "real training feature set."
        )