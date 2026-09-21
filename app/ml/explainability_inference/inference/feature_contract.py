from __future__ import annotations

import pandas as pd

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

ID_COLUMN = "customer_unique_id"


class FeatureValidationError(ValueError):
    """Raised when input features do not satisfy the model's feature contract."""


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
   
    if df is None or len(df) == 0:
        raise FeatureValidationError("Input feature data is empty.")

    if ID_COLUMN not in df.columns:
        raise FeatureValidationError(
            f"Missing required identifier column: '{ID_COLUMN}'."
        )

    missing = [f for f in REQUIRED_FEATURES if f not in df.columns]
    if missing:
        raise FeatureValidationError(
            f"Missing required feature column(s): {missing}"
        )

    null_counts = df[REQUIRED_FEATURES].isnull().sum()
    bad_columns = null_counts[null_counts > 0]
    if not bad_columns.empty:
        raise FeatureValidationError(
            f"Required feature(s) contain null values: {bad_columns.to_dict()}"
        )

    non_numeric = [
        col
        for col in REQUIRED_FEATURES
        if FEATURE_DTYPES.get(col) == "numeric"
        and not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise FeatureValidationError(
            f"Feature(s) expected to be numeric but are not: {non_numeric}"
        )

    ordered_columns = [ID_COLUMN] + REQUIRED_FEATURES
    validated = df[ordered_columns].copy()
    return validated


def get_model_feature_names(model) -> list[str] | None:
   
    feature_names = getattr(model, "feature_name_", None)
    if feature_names is not None:
        return list(feature_names)

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return list(feature_names)

    return None


def _extract_feature_names(model_or_artifact) -> list[str] | None:

    if isinstance(model_or_artifact, dict):
        if "feature_cols" in model_or_artifact:
            return list(model_or_artifact["feature_cols"])
        model_or_artifact = model_or_artifact.get("model", model_or_artifact)

    return get_model_feature_names(model_or_artifact)


def check_contract_matches_model(model) -> None:

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
