#!/usr/bin/env python3
"""
run_all.py

WHY THIS FILE EXISTS
---------------------
Orchestrates the entire app/ml training + calibration pipeline end to
end, from the Logistic Regression baseline through the final calibrated
artifact — mirrors scripts/run_pipeline.py's own style (one subprocess
per step, fail-fast, clear banners) for the ml side of the repo, the
same way run_pipeline.py already does for the Feature Engineering side.

Requires the Feature Engineering pipeline (scripts/run_pipeline.py) to
have already been run at least once — every step here reads from
model_ready_train/val/test and features_encoded, which run_pipeline.py
produces.

WHAT IT RUNS, IN ORDER
------------------------
1. logistic_regression.py   (Person 1: baseline model)
2. train_lightgbm_model.py  (Person 2: LightGBM)
3. imbalance_experiments.py (Person 3: class-imbalance comparison — informational
                              only, nothing downstream reads its output; skip with
                              --skip-imbalance if you just want a working model fast)
4. threshold_analysis.py    (Person 4: decision-threshold sweep — needs steps 1+2)
5. model_comparison.py      (Person 4: LR vs LightGBM comparison — needs steps 1+2)
6. run_calibration.py       (Person 5: calibrates LightGBM, the FINAL deployable
                              artifact — needs step 2, reads step 4's recommended
                              tuning metric if present)

Each step runs as its own SEPARATE subprocess (python -m app.ml.<name>,
or, for the last step, `python app/ml/run_calibration.py` — deliberately
NOT `python -m app.ml.calibration` directly, which has a known joblib/
pickle bug; see run_calibration.py's own docstring for why). Subprocess
isolation means one step's global state (matplotlib's backend, warnings
filters, optuna's verbosity setting, etc.) can never leak into another
step, and a step that calls sys.exit() on its own failure only
terminates itself — this script catches that via the return code and
stops the whole run with a clear message, it doesn't die uninformatively.

WHAT IT DOES NOT DO
---------------------
Doesn't touch MySQL schema, the Feature Engineering pipeline, or
app/api — purely orchestrates the app/ml/*.py training scripts already
in this repo. Doesn't set MODEL_PATH/MODEL_VERSION in .env for you
afterward — see the final printed message for the exact path to use.

USAGE
-----
    python app/ml/run_all.py
    python app/ml/run_all.py --skip-imbalance   # skip the slow, optional step 3
    python app/ml/run_all.py --skip-training    # only run steps 4-6, if 1+2 already ran

(from the repo root, same convention as scripts/run_pipeline.py)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# This file lives at app/ml/run_all.py, so parents[2] is the repo root.
ROOT = Path(__file__).resolve().parents[2]


def get_python_executable() -> str:
    """Use the currently running Python interpreter if it's already inside a virtualenv,
    or find the project-local virtual environment."""
    # If already running inside a venv, reuse that exact interpreter
    if sys.prefix != sys.base_prefix:
        return sys.executable

    # Check project-root .venv first
    project_venv = ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    if project_venv.exists():
        return str(project_venv)

    # Check parent folder .venv if any
    parent_venv = ROOT.parent / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    if parent_venv.exists():
        return str(parent_venv)

    return sys.executable


def run_step(name: str, *args: str) -> None:
    command = [get_python_executable(), *args]
    print(f"\n{'=' * 70}")
    print(f"=== {name} ===")
    print(f"{'=' * 70}")
    print("Running:", " ".join(str(part) for part in command))
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"\n{name} failed with exit code {result.returncode}. Stopping.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full app/ml training + calibration pipeline end to end."
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip steps 1-3 (logistic regression, LightGBM, imbalance experiments) "
             "-- use if those artifacts already exist and you only want threshold "
             "analysis, comparison, and calibration to (re-)run.",
    )
    parser.add_argument(
        "--skip-imbalance",
        action="store_true",
        help="Skip step 3 (imbalance experiments) -- it's informational only, "
             "nothing downstream depends on its output.",
    )
    args = parser.parse_args()

    if not args.skip_training:
        run_step("1/6 Logistic Regression (Person 1)", "-m", "app.ml.logistic_regression")
        run_step("2/6 LightGBM (Person 2)", "-m", "app.ml.train_lightgbm_model")
        if not args.skip_imbalance:
            run_step("3/6 Imbalance experiments (Person 3)", "-m", "app.ml.imbalance_experiments")
        else:
            print("\nSkipping step 3/6 (imbalance experiments) -- informational only.")
    else:
        print("\nSkipping steps 1-3 (--skip-training) -- assuming their artifacts already exist.")

    run_step("4/6 Threshold analysis (Person 4)", "-m", "app.ml.threshold_analysis")
    run_step("5/6 Model comparison (Person 4)", "-m", "app.ml.model_comparison")
    run_step("6/6 Calibration (Person 5)", str(ROOT / "app" / "ml" / "run_calibration.py"))

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("Final calibrated artifact: outputs/models/lgb_churn_model_calibrated.joblib")
    print("Point MODEL_PATH in .env at it (and set MODEL_VERSION) to serve it via the API.")


if __name__ == "__main__":
    main()