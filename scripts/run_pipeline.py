#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "app" / "Database"
FEATURES_DIR = ROOT / "app" / "Features"


def get_python_executable() -> str:
    venv_python = ROOT.parent / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    if venv_python.exists():
        return str(venv_python)
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

    print("\nPipeline complete.")
    print("Model-ready tables created in MySQL:")
    print("- model_ready_train")
    print("- model_ready_val")
    print("- model_ready_test")


if __name__ == "__main__":
    main()
