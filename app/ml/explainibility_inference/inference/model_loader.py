from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


class ModelArtifactNotFoundError(FileNotFoundError):
    pass

def load_model(path: str | Path | None) -> Any:

    if path is None:
        raise ModelArtifactNotFoundError(
            "No model path given (received None). Set MODEL_PATH in .env "
            "or pass an explicit path — there is no dummy model to fall "
            "back to."
        )

    artifact_path = Path(path)

    if not artifact_path.exists():
        raise ModelArtifactNotFoundError(
            f"Model artifact not found at '{artifact_path}'. "
            "Make sure the model has been trained/saved before running "
            "inference, or check that the path is correct."
        )

    try:
        model = joblib.load(artifact_path)
    except Exception as exc:  # noqa: BLE001 - want a clear, wrapped error
        raise RuntimeError(
            f"Failed to load model artifact at '{artifact_path}': {exc}"
        ) from exc

    return model
