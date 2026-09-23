
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