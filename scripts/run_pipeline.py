#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "app" / "Database"
FEATURES_DIR = ROOT / "app" / "Features"
TRENDS_DIR = ROOT / "app" / "segmentation" / "customer_analytics" / "Trends"
ANALYTICS_DIR = ROOT / "app" / "segmentation" / "customer_analytics"
FORECASTING_DIR = ANALYTICS_DIR / "forecasting"
ML_DIR = ROOT / "app" / "ml"

# --- Modules that must be run with `python -m ...` ---
# These scripts do `from app.X import Y` (absolute, package-style imports)
# with no sys.path bootstrap of their own, so invoking them as a direct
# file path (`python /abs/path/script.py`) fails with
# "ModuleNotFoundError: No module named 'app'" -- only the script's own
# folder ends up on sys.path, not the project root. `-m` puts the
# current working directory (the repo root, since we set cwd=ROOT) on
# sys.path instead, which is also how the README documents running
# each of these.
COHORT_MODULE = "app.segmentation.customer_analytics.cohort_analysis"
FORECASTING_MODULE = "app.segmentation.customer_analytics.forecasting.forecasting"
PREDICTIONS_MODULE = "app.ml.explainibility_inference.inference.generate_predictions_table"

# --- Segmentation step ---
# Order matters: risk tier + GMM segmentation and CLV both feed the
# campaign recommendations step, so campaign recommendations must run
# last. All three require churn_predictions (written by the ML stage)
# to exist.
SEGMENTATION_MODULE = "app.segmentation.customer_intelligence.pipeline.generate_customer_intelligence_tables"
CLV_MODULE = "app.segmentation.customer_intelligence.generate_clv_table"
CAMPAIGN_MODULE = "app.segmentation.customer_intelligence.campaign_engine.pipeline.generate_campaign_recommendations_table"


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
    """Run a script by direct file path. Only safe for scripts that either
    bootstrap their own sys.path or don't need the `app` package at all."""
    command = [get_python_executable(), str(script_path), *args]
    print(f"\n=== {name} ===")
    print("Running:", " ".join(str(part) for part in command))
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {result.returncode}")


def run_module_step(name: str, module: str, *args: str) -> None:
    """Run a script as a `-m` module so its `from app.X import Y` imports
    resolve correctly regardless of its own folder location."""
    command = [get_python_executable(), "-m", module, *args]
    print(f"\n=== {name} ===")
    print("Running:", " ".join(str(part) for part in command))
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full customer intelligence pipeline end to end: "
                     "ingestion -> feature engineering -> ML -> segmentation."
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip the CSV-to-MySQL ingestion step if data is already loaded.",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Skip generating the customer churn predictions table.",
    )
    parser.add_argument(
        "--skip-cohort",
        action="store_true",
        help="Skip cohort analysis.",
    )
    parser.add_argument(
        "--skip-trends",
        action="store_true",
        help="Skip historical trend analysis.",
    )
    parser.add_argument(
        "--skip-forecasting",
        action="store_true",
        help="Skip historical trend forecasting.",
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Skip Risk Tier Classification + GMM Segmentation.",
    )
    parser.add_argument(
        "--skip-clv",
        action="store_true",
        help="Skip Customer Lifetime Value calculation.",
    )
    parser.add_argument(
        "--skip-campaign",
        action="store_true",
        help="Skip Campaign Recommendations. Requires segmentation and CLV "
             "tables to already exist if you skip those steps individually.",
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
             "comparison, and calibration.",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # 1. Data ingestion
    # ---------------------------------------------------------------
    if not args.skip_ingest:
        run_step("Ingesting source data", DB_DIR / "ingest.py", "all", "--replace")

    # ---------------------------------------------------------------
    # 2. Feature engineering
    # ---------------------------------------------------------------
    run_step("Building feature tables", FEATURES_DIR / "build_features.py")
    run_step("Building churn labels", FEATURES_DIR / "build_churn_label.py")
    run_step("Merging features and labels", FEATURES_DIR / "merge.py")
    run_step("Encoding and transforming features", FEATURES_DIR / "encoding_transformation.py")
    run_step(
        "Selecting features and scaling",
        FEATURES_DIR / "feature_selection_and_scaling.py",
    )

    # ---------------------------------------------------------------
    # 3. Machine learning (training + churn predictions)
    # ---------------------------------------------------------------
    if args.train_all:
        run_step(
            "Running complete ML training and calibration pipeline",
            ML_DIR / "run_all.py",
        )
    else:
        if args.train_logistic:
            run_step("Training Logistic Regression model", ML_DIR / "logistic_regression.py")

        if args.train_lightgbm:
            run_step("Training LightGBM model", ML_DIR / "train_lightgbm_model.py")

    if not args.skip_predictions:
        run_module_step("Generating customer churn predictions", PREDICTIONS_MODULE)

    # ---------------------------------------------------------------
    # 4. Segmentation -- customer_analytics
    # ---------------------------------------------------------------
    if not args.skip_cohort:
        run_module_step("Customer Cohort Analysis", COHORT_MODULE)

    if not args.skip_trends:
        run_step(
            "Historical Trend Analysis",
            TRENDS_DIR / "pipeline.py",
        )

    if not args.skip_forecasting:
        run_module_step("Historical Trend Forecasting", FORECASTING_MODULE)

    # ---------------------------------------------------------------
    # 5. Segmentation -- customer_intelligence
    # ---------------------------------------------------------------
    if not args.skip_segmentation:
        run_module_step(
            "Risk Tier Classification + GMM Segmentation",
            SEGMENTATION_MODULE,
        )

    if not args.skip_clv:
        run_module_step("Customer Lifetime Value", CLV_MODULE)

    if not args.skip_campaign:
        run_module_step("Campaign Recommendations", CAMPAIGN_MODULE)

    print("\nPipeline complete.")
    print("Tables created/refreshed in MySQL:")
    print("- model_ready_train")
    print("- model_ready_val")
    print("- model_ready_test")
    print("- churn_predictions")
    print("- customer_cohort_analysis")
    print("- historical_revenue_trend / historical_churn_trend / historical_trend_combined")
    print("- forecast_revenue_trend / forecast_churn_trend / forecast_trend_combined")
    print("- customer_intelligence_base")
    print("- customer_risk_tiers")
    print("- customer_segments")
    print("- customer_clv")
    print("- customer_campaign_recommendations")


if __name__ == "__main__":
    main()
