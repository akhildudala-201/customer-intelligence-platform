"""
app/segmentation/customer_analytics/Trends/export.py

Persistence and export module for Historical Trend Analysis (Person 5).
Handles saving Schema 6.5 tables into MySQL and exporting JSON/CSV reports.
"""

import json
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from sqlalchemy import engine as sql_engine


def _find_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "app").exists() and (parent / "data").exists():
            return parent
    return Path(__file__).resolve().parents[4]


ROOT = _find_root()
OUTPUT_DIR = ROOT / "outputs" / "reports"


class TrendExporter:
    """
    Persists trend analysis tables to MySQL and exports analytical reports.
    """

    def __init__(
        self,
        db_engine: Optional[sql_engine.Engine] = None,
        output_dir: Optional[Path] = None,
    ):
        self.db_engine = db_engine or self._init_db_engine()
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _init_db_engine(self):
        try:
            from app.Database.database import engine
            return engine
        except Exception:
            try:
                from Database.database import engine
                return engine
            except Exception:
                return None

    def save_to_database(
        self,
        trend_results: Dict[str, Dict[str, pd.DataFrame]],
        table_prefix: str = "historical_",
    ) -> Dict[str, int]:
        """
        Save revenue, churn, and combined master trends into MySQL tables.
        """
        if self.db_engine is None:
            print("[TrendExporter] No DB engine configured. Skipping MySQL persistence.")
            return {}

        combined_bundle = trend_results.get("all_granularities_combined", {})
        rev_df = combined_bundle.get("revenue")
        churn_df = combined_bundle.get("churn")
        master_df = combined_bundle.get("master")

        row_counts = {}

        with self.db_engine.begin() as conn:
            if rev_df is not None and not rev_df.empty:
                rev_table = f"{table_prefix}revenue_trend"
                rev_df.to_sql(rev_table, con=conn, if_exists="replace", index=False, chunksize=2000)
                row_counts[rev_table] = len(rev_df)
                print(f"[TrendExporter] Saved {len(rev_df):,} rows -> MySQL table `{rev_table}`")

            if churn_df is not None and not churn_df.empty:
                churn_table = f"{table_prefix}churn_trend"
                churn_df.to_sql(churn_table, con=conn, if_exists="replace", index=False, chunksize=2000)
                row_counts[churn_table] = len(churn_df)
                print(f"[TrendExporter] Saved {len(churn_df):,} rows -> MySQL table `{churn_table}`")

            if master_df is not None and not master_df.empty:
                master_table = f"{table_prefix}trend_combined"
                master_df.to_sql(master_table, con=conn, if_exists="replace", index=False, chunksize=2000)
                row_counts[master_table] = len(master_df)
                print(f"[TrendExporter] Saved {len(master_df):,} rows -> MySQL table `{master_table}`")

        return row_counts

    def export_reports(
        self,
        trend_results: Dict[str, Dict[str, pd.DataFrame]],
        export_csv: bool = True,
        export_json: bool = True,
    ) -> Dict[str, str]:
        """
        Export trend datasets and a summary JSON report to disk.
        """
        exported_files = {}

        for gran in ("daily", "weekly", "monthly"):
            if gran in trend_results:
                bundle = trend_results[gran]
                rev_df = bundle.get("revenue")
                churn_df = bundle.get("churn")
                combined_df = bundle.get("combined")

                if export_csv:
                    if rev_df is not None and not rev_df.empty:
                        rev_path = self.output_dir / f"historical_revenue_trend_{gran}.csv"
                        rev_df.to_csv(rev_path, index=False)
                        exported_files[f"revenue_{gran}_csv"] = str(rev_path)

                    if churn_df is not None and not churn_df.empty:
                        churn_path = self.output_dir / f"historical_churn_trend_{gran}.csv"
                        churn_df.to_csv(churn_path, index=False)
                        exported_files[f"churn_{gran}_csv"] = str(churn_path)

                    if combined_df is not None and not combined_df.empty:
                        comb_path = self.output_dir / f"historical_trend_combined_{gran}.csv"
                        combined_df.to_csv(comb_path, index=False)
                        exported_files[f"combined_{gran}_csv"] = str(comb_path)

        # Export master dataset with all granularities
        all_master = trend_results.get("all_granularities_combined", {}).get("master")
        if all_master is not None and not all_master.empty:
            master_path = self.output_dir / "historical_trend_master_schema_6_5.csv"
            all_master.to_csv(master_path, index=False)
            exported_files["master_schema_6_5_csv"] = str(master_path)

        # Generate summary metadata JSON
        if export_json:
            summary = self._generate_summary_metadata(trend_results)
            json_path = self.output_dir / "trend_summary_report.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            exported_files["summary_json"] = str(json_path)

        return exported_files

    def _generate_summary_metadata(
        self, trend_results: Dict[str, Dict[str, pd.DataFrame]]
    ) -> dict:
        """Construct a high-level summary of historical trend metrics with validation results."""
        summary = {
            "schema_version": "6.5",
            "component": "Person 5 - Historical Trend Analysis",
            "target_consumer": "Person 6 - Forecasting",
            "validation_results": trend_results.get("all_granularities_combined", {}).get("validation", {}),
            "granularities": {},
        }

        for gran in ("daily", "weekly", "monthly"):
            if gran in trend_results:
                bundle = trend_results[gran]
                rev = bundle.get("revenue", pd.DataFrame())
                churn = bundle.get("churn", pd.DataFrame())

                censored_count = (
                    int(churn["is_censored"].sum())
                    if not churn.empty and "is_censored" in churn.columns
                    else 0
                )

                summary["granularities"][gran] = {
                    "total_periods": len(rev),
                    "date_start": str(rev["period_date"].min()) if not rev.empty else None,
                    "date_end": str(rev["period_date"].max()) if not rev.empty else None,
                    "censored_periods": censored_count,
                    "total_historical_revenue": float(rev["total_revenue"].sum()) if not rev.empty else 0.0,
                    "total_orders": int(rev["order_count"].sum()) if not rev.empty else 0,
                    "avg_period_revenue": float(rev["total_revenue"].mean()) if not rev.empty else 0.0,
                    "overall_avg_order_value": float(rev["avg_order_value"].mean()) if not rev.empty else 0.0,
                    "total_churned_recorded": int(churn["churned_customers"].sum()) if not churn.empty else 0,
                    "avg_period_churn_rate": float(churn["churn_rate"].mean()) if not churn.empty else 0.0,
                }

        return summary
