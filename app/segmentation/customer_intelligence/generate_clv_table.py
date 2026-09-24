from __future__ import annotations

import pandas as pd

from app.Database.database import engine
from app.segmentation.customer_intelligence.clv_calculator import (
    CLV_COLUMN,
    ID_COLUMN,
    LIFESPAN_COLUMN,
    PURCHASE_FREQUENCY_COLUMN,
    VALUE_TIER_COLUMN,
    compute_customer_clv,
)

CLV_TABLE = "customer_clv"

OUTPUT_COLUMNS = [
    ID_COLUMN,
    PURCHASE_FREQUENCY_COLUMN,
    LIFESPAN_COLUMN,
    CLV_COLUMN,
    VALUE_TIER_COLUMN,
]


def generate_clv_table(**compute_kwargs) -> pd.DataFrame:

    full = compute_customer_clv(**compute_kwargs)

    missing = set(OUTPUT_COLUMNS) - set(full.columns)
    if missing:
        raise ValueError(
            f"compute_customer_clv() output is missing expected column(s) "
            f"{missing} -- did clv_calculator.py's column names change "
            "without updating OUTPUT_COLUMNS here?"
        )

    output = full[OUTPUT_COLUMNS].copy()

    output.to_sql(CLV_TABLE, con=engine, if_exists="replace", index=False)
    print(f"Wrote {len(output)} row(s) to `{CLV_TABLE}`.", flush=True)

    return output


def _print_summary(output: pd.DataFrame) -> None:

    print("\n--- customer_clv: first 10 rows ---", flush=True)
    print(output.head(10).to_string(index=False), flush=True)

    print("\n--- value_tier counts ---", flush=True)
    print(output[VALUE_TIER_COLUMN].value_counts().to_string(), flush=True)

    print("\n--- clv summary stats ---", flush=True)
    print(output[CLV_COLUMN].describe().to_string(), flush=True)


if __name__ == "__main__":
    result = generate_clv_table()
    _print_summary(result)
