#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "app" / "Database"
FEATURES_DIR = ROOT / "app" / "Features"
ML_DIR = ROOT / "app" / "ml"


def get_python_executable() -> str:
    """Use the currently running Python interpreter if it's already inside a virtualenv,
    or find the project-local virtual environment."""
    if sys.prefix != sys.base_prefix:
        return sys.executable

    project_venv = ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    if project_venv.exists():
        return str(project_venv)

    parent_venv = ROOT.parent / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    if parent_venv.exists():
        return str(parent_venv)

    return sys.executable


def run_step(name: str, script_path: Path, *args: str) -> None:
    command = [get_python_executable(), str(script_path), *args]
    print(f"\n=== {name} ===")
    print("Running:", " ".join(str(part) for part in command))
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full customer intelligence ML pipeline and push model-ready tables to MySQL."
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip the CSV-to-MySQL ingestion step if data is already loaded.",
    )
    
    parser.add_argument(
        "--train-logistic",
        action="store_true",
        help="Train the Logistic Regression churn model after building model-ready tables.",
    )
    parser.add_argument(
        "--train-lightgbm",
        action="store_true",
        help="Train the LightGBM churn model after building model-ready tables.",
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Run the complete ML pipeline after building model-ready tables: "
             "training, imbalance experiments, threshold analysis, model "
             "comparison, calibration, and churn_predictions refresh.",
    )
    args = parser.parse_args()

    if not args.skip_ingest:
        run_step("Ingesting source data", DB_DIR / "ingest.py", "all", "--replace")

    run_step("Building feature tables", FEATURES_DIR / "build_features.py")
    run_step("Building churn labels", FEATURES_DIR / "build_churn_label.py")
    run_step("Merging features and labels", FEATURES_DIR / "merge.py")
    run_step("Encoding and transforming features", FEATURES_DIR / "encoding_transformation.py")
    run_step(
        "Selecting features and scaling",
        FEATURES_DIR / "feature_selection_and_scaling.py",
    )

    if args.train_all:
        run_step(
            "Running complete ML training and calibration pipeline",
            ML_DIR / "run_all.py",
        )
        run_step(
            "Refreshing churn predictions table",
            ML_DIR / "explainibility_interface"
            / "inference"
            / "generate_predictions_table.py",
        )
    else:
        if args.train_logistic:
            run_step("Training Logistic Regression model", ML_DIR / "logistic_regression.py")

        if args.train_lightgbm:
            run_step("Training LightGBM model", ML_DIR / "train_lightgbm_model.py")

    print("\nPipeline complete.")
    print("Model-ready tables created in MySQL:")
    print("- model_ready_train")
    print("- model_ready_val")
    print("- model_ready_test")
    if args.train_all:
        print("- churn_predictions (refreshed by generate_predictions_table.py)")


if __name__ == "__main__":
    main()
