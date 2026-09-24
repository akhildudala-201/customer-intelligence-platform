from __future__ import annotations

import pandas as pd

from app.Database.database import engine
from app.segmentation.customer_intelligence.data_access.campaign_input_repository import (
    build_campaign_input_base,
)
from app.segmentation.customer_intelligence.campaign_engine.recommender import (
    CAMPAIGN_NAME_COLUMN,
    OUTPUT_COLUMNS,
    REASON_CODE_COLUMN,
    recommend_campaigns,
)

CAMPAIGN_RECOMMENDATIONS_TABLE = "customer_campaign_recommendations"


def generate_campaign_recommendations_table() -> pd.DataFrame:
   
    base = build_campaign_input_base()
    print(f"  -> {len(base)} customer(s) have risk_tier + segment + value_tier.")

    recommendations = recommend_campaigns(base)

    missing = set(OUTPUT_COLUMNS) - set(recommendations.columns)
    if missing:
        raise ValueError(
            f"recommend_campaigns() output is missing expected column(s) "
            f"{missing} -- did recommender.py's column names change without "
            "updating OUTPUT_COLUMNS there?"
        )

    output = recommendations[OUTPUT_COLUMNS].copy()
    output.to_sql(CAMPAIGN_RECOMMENDATIONS_TABLE, con=engine, if_exists="replace", index=False)
    print(f"Wrote {len(output)} row(s) to `{CAMPAIGN_RECOMMENDATIONS_TABLE}`.", flush=True)

    return output


def _print_summary(output: pd.DataFrame) -> None:

    print("\n--- customer_campaign_recommendations: first 10 rows ---", flush=True)
    print(output.head(10).to_string(index=False), flush=True)

    print("\n--- campaign_priority counts ---", flush=True)
    print(output["campaign_priority"].value_counts().sort_index().to_string(), flush=True)

    print("\n--- reason_code counts ---", flush=True)
    print(output[REASON_CODE_COLUMN].value_counts().to_string(), flush=True)

    print("\n--- campaign_name counts ---", flush=True)
    print(output[CAMPAIGN_NAME_COLUMN].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    result = generate_campaign_recommendations_table()
    _print_summary(result)
