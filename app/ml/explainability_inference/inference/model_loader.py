"""
model_loader.py

WHY THIS FILE EXISTS
---------------------
Loading a model artifact from disk is a distinct concern from using that
model to predict. Keeping it in its own module means the rest of the
pipeline never touches file paths or joblib directly — swapping the
artifact file only requires a different `path` argument, not a code
change here.

WHAT IT ACCEPTS
---------------
`load_model(path)` accepts a string or Path to a .joblib file (or None,
which raises a clear error rather than crashing on a bad Path() call).

WHAT IT RETURNS
----------------
The deserialized Python object stored in that joblib file. This function
does NOT know or care what type of object it is (a bare model, or a dict
artifact like {"model": ..., "feature_cols": ...}) — that is the
model_adapter's job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


class ModelArtifactNotFoundError(FileNotFoundError):
    """Raised when the model artifact file does not exist on disk."""


def load_model(path: str | Path | None) -> Any:
    """
    Load a joblib-serialized model artifact from `path`.

    Raises:
        ModelArtifactNotFoundError: if path is None or the file does not exist.
        RuntimeError: if the file exists but fails to deserialize.

    Returns:
        The deserialized model object (whatever was saved to disk).
    """
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