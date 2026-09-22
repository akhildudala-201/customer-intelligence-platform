from __future__ import annotations

import pandas as pd
from sqlalchemy import bindparam, text

from app.ml.explainibility_inference.inference.feature_contract import ID_COLUMN, REQUIRED_FEATURES

TABLE_NAME = "features_encoded"


class CustomerNotFoundError(LookupError):
   
def _get_engine():

    from app.Database.database import engine

    return engine


def _select_columns() -> list[str]:

    return [ID_COLUMN] + REQUIRED_FEATURES


def get_customer_features(customer_unique_id: str) -> pd.DataFrame:

    return get_customer_features_batch([customer_unique_id])


def get_customer_features_batch(customer_unique_ids: list[str]) -> pd.DataFrame:

    if not customer_unique_ids:
        raise ValueError("customer_unique_ids must contain at least one id.")

    columns = _select_columns()
    column_list_sql = ", ".join(columns)

    query = text(
        f"SELECT {column_list_sql} FROM {TABLE_NAME} "  # nosec - table/column
        f"WHERE {ID_COLUMN} IN :ids"                     # names are code
    ).bindparams(bindparam("ids", expanding=True))

    engine = _get_engine()
    with engine.connect() as connection:
        result = pd.read_sql(
            query,
            connection,
            params={"ids": list(customer_unique_ids)},
        )

    if result.empty:
        raise CustomerNotFoundError(
            f"No rows found in `{TABLE_NAME}` for customer_unique_id(s): "
            f"{customer_unique_ids}"
        )

    return result[columns].reset_index(drop=True)


def get_all_customer_features() -> pd.DataFrame:

    columns = _select_columns()
    column_list_sql = ", ".join(columns)

    query = text(f"SELECT {column_list_sql} FROM {TABLE_NAME}")  # nosec - table/column names are code

    engine = _get_engine()
    with engine.connect() as connection:
        result = pd.read_sql(query, connection)

    if result.empty:
        raise CustomerNotFoundError(f"`{TABLE_NAME}` has no rows to score.")

    return result[columns].reset_index(drop=True)
