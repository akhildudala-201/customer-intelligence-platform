"""
app/Trends/pipeline.py

Pipeline Runner for Person 5 — Historical Trend Analysis (Schema 6.5).
Executes data loading, daily/weekly/monthly revenue and churn trend aggregation,
Schema 6.5 validation, persistence to MySQL, and analytical report generation.
"""

import argparse
from pathlib import Path
import sys

def _find_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "app").exists() and (parent / "data").exists():
            return parent
    return Path(__file__).resolve().parents[4]


ROOT = _find_root()
APP_DIR = ROOT / "app"
TRENDS_DIR = Path(__file__).resolve().parent
for path in (str(ROOT), str(APP_DIR), str(TRENDS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from app.segmentation.customer_analytics.Trends.data_loader import TrendDataLoader
    from app.segmentation.customer_analytics.Trends.export import TrendExporter
    from app.segmentation.customer_analytics.Trends.trend_engine import HistoricalTrendEngine
except (ImportError, ModuleNotFoundError):
    from data_loader import TrendDataLoader
    from export import TrendExporter
    from trend_engine import HistoricalTrendEngine


def run_trend_pipeline(
    granularity: str = "all",
    save_db: bool = True,
    output_dir: Path = ROOT / "outputs" / "reports",
    export_files: bool = True,
) -> dict:
    """
    Execute end-to-end Historical Trend Analysis pipeline.
    """
    print("\n" + "=" * 70)
    print("PERSON 5 — HISTORICAL TREND ANALYSIS PIPELINE (SCHEMA 6.5)")
    print("=" * 70)

    # 1. Load Data
    print("\n[1/4] Loading transaction and churn datasets...")
    loader = TrendDataLoader()
    orders_df = loader.load_transaction_data()
    print(f"      Loaded {len(orders_df):,} transaction records.")
    
    churn_df = loader.load_customer_churn_data()
    print(f"      Loaded {len(churn_df):,} customer churn records.")

    # 2. Compute Trends
    print("\n[2/4] Aggregating historical Revenue and Churn trends...")
    engine = HistoricalTrendEngine()
    grans = ["daily", "weekly", "monthly"] if granularity == "all" else [granularity]
    trend_results = engine.generate_all_trends(orders_df, churn_df, granularities=grans)

    for g in grans:
        rev_count = len(trend_results[g]["revenue"])
        churn_count = len(trend_results[g]["churn"])
        print(f"      -> {g.capitalize()} level: {rev_count} revenue periods, {churn_count} churn periods computed.")

    # 3. Save to MySQL
    persisted_tables = {}
    if save_db:
        print("\n[3/4] Persisting Schema 6.5 tables to MySQL database...")
        exporter = TrendExporter(output_dir=output_dir)
        persisted_tables = exporter.save_to_database(trend_results)
    else:
        print("\n[3/4] Skipping MySQL database persistence (--no-db flag).")

    # 4. Export CSV / JSON Reports
    exported_files = {}
    if export_files:
        print(f"\n[4/4] Exporting trend reports to: {output_dir}")
        exporter = TrendExporter(output_dir=output_dir)
        exported_files = exporter.export_reports(trend_results)
        for key, filepath in exported_files.items():
            print(f"      Exported: {Path(filepath).name}")
    else:
        print("\n[4/4] Skipping file export.")

    print("\n" + "=" * 70)
    print("HISTORICAL TREND ANALYSIS COMPLETED SUCCESSFULLY")
    print("Ready to feed Person 6 (Forecasting - Schema 6.5)")
    print("=" * 70 + "\n")

    return {
        "status": "success",
        "persisted_tables": persisted_tables,
        "exported_files": exported_files,
        "granularities": grans,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Person 5: Historical Trend Analysis CLI (Schema 6.5)"
    )
    parser.add_argument(
        "--granularity",
        choices=["daily", "weekly", "monthly", "all"],
        default="all",
        help="Time granularity to compute trends for (default: all)",
    )
    parser.add_argument(
        "--save-db",
        action="store_true",
        default=True,
        help="Persist tables to MySQL (default: True)",
    )
    parser.add_argument(
        "--no-db",
        action="store_false",
        dest="save_db",
        help="Do not persist tables to MySQL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "reports",
        help="Directory to save CSV/JSON reports",
    )
    parser.add_argument(
        "--no-export",
        action="store_false",
        dest="export_files",
        default=True,
        help="Disable file export",
    )

    args = parser.parse_args()
    run_trend_pipeline(
        granularity=args.granularity,
        save_db=args.save_db,
        output_dir=args.output_dir,
        export_files=args.export_files,
    )


if __name__ == "__main__":
    main()
