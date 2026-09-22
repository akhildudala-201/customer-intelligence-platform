from __future__ import annotations

from pathlib import Path

import yaml


def load_reason_codes(path: str | Path) -> dict[str, dict]:

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
