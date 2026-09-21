# Explainability & Inference

The folder is part of the project: it takes a trained churn model
and serves predictions from it — one customer at a time, in batch, or
as a full table for the segmentation part.

Note: this branch only has shap and interface related codes. The rest of the project (data
ingestion, feature engineering, model training, calibration) is on
other branches / other people's work, and this code depends on that
being merged in before it can actually run.

## Prerequisites

Before this code will run, you need:

1. **Python 3.12**
2. **The rest of the repo merged in** — this folder alone can't run by
   itself. It needs:
   - `app/Database/` (database connection code)
   - `app/Features/` (builds the `features_encoded` table this reads from)
   - `app/ml/train_lightgbm.py` (to tain the model)
   - `app/ml/calibration.py` + `app/ml/run_calibration.py` (produces the
     trained model file this loads)
3. **A MySQL database** already set up and reachable, with the
   `features_encoded` table populated (built by the Feature Engineering
   part above)
4. **A trained, calibrated model file** (a `.joblib` file produced by
   `app/ml/calibration.py`)
5. **A `.env` file** in the repo root with:
   ```
   DB_HOST=...
   DB_PORT=...
   DB_USER=...
   DB_PASSWORD=...
   DB_NAME=...
   MODEL_PATH=path/to/your/calibrated_model.joblib
   MODEL_VERSION=whatever-you-want-to-call-it
   ```

## Setup

```bash
pip install -r app/ml/explainability_inference/requirements.txt
```

## What's in this folder

```
data_access/   → reads customer data from the database
config/        → maps model output to human-readable reason codes
explainability/→ SHAP — explains why a customer got their score
inference/     → loads the model and runs predictions
api/           → a web API to request predictions on demand
smoke_check.py → quick manual check that everything works
requirements.txt → just the packages this folder needs
```

Tests for this folder live in `app/ml/tests/explainability_inference_tests/`.

## Running it

**Score every customer at once** (this is what the segmentation part reads from):
```bash
python -m app.ml.explainability_inference.inference.generate_predictions_table
```

**Or run a live API** to get predictions on demand:
```bash
uvicorn app.ml.explainability_inference.api.main:app --reload --host 127.0.0.1 --port 8000
```

**Quick check that the pipeline works** (prints one prediction, no setup needed beyond the prerequisites above):
```bash
python -m app.ml.explainability_inference.smoke_check
```

## Running the tests

```bash
python -m pytest app/ml/tests/explainability_inference_tests -v
```

