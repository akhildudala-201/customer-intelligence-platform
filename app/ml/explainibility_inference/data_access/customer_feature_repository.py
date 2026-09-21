"""
customer_feature_repository.py

WHY THIS FILE EXISTS
---------------------
`ChurnPredictor.predict()` needs customer feature rows as a pandas
DataFrame. Until now the only source was hand-built test data. This
module is the read-only bridge to the `features_encoded` MySQL table —
the encoded/scaled output of the Feature Engineering pipeline
(app/Features/encoding_transformation.py, feature_selection_and_scaling.py)
— NOT the raw `customer_features` table, which lacks the encoded columns
this model needs (see feature_contract.py).

It is READ-ONLY and reuses the existing `app.Database.database` engine
instead of opening a new connection, duplicating credential handling, or
creating a second DB module. It never CREATEs, ALTERs, INSERTs, UPDATEs
or DELETEs anything — only SELECTs the columns feature_contract.py
already declares as required. Nothing about the database, its schema, or
`app/Database/database.py` is touched or changed by this file.

WHAT IT ACCEPTS
---------------
`get_customer_features(customer_unique_id)` - a single customer id (str).
`get_customer_features_batch(customer_unique_ids)` - a list of ids.

WHAT IT RETURNS
----------------
A pandas DataFrame with ID_COLUMN + REQUIRED_FEATURES columns — exactly
the columns `feature_contract.validate_features()` expects — ready to
pass straight into `ChurnPredictor.predict()` with no reshaping. Column
names/order are pulled from feature_contract.py at query time, so this
file has no hard-coded duplicate of the feature list.

Raises CustomerNotFoundError if none of the requested id(s) exist in the
table, so callers (e.g. the API layer) can turn that into a clean 404
instead of an opaque "input is empty" error from feature validation.

WHICH PART IS DUMMY / REPLACEABLE
-----------------------------------
None of this is dummy — it queries the real `features_encoded` table
using the real feature contract names (see feature_contract.py). Note,
however, that feature_contract.REQUIRED_FEATURES itself currently has an
unresolved disagreement with Persons 1/2's training code over 5 of those
17 column names (see the "UNRESOLVED" note at the top of
feature_contract.py) — this file will faithfully SELECT whatever that
list says, correct or not.

WHAT CHANGES WHEN PERSON 5 SUPPLIES THE FINAL MODEL
-----------------------------------------------------
Nothing here directly, as long as `features_encoded` remains the table
holding live, per-customer encoded feature values. If
feature_contract.REQUIRED_FEATURES is revised (see the unresolved note
above) once a real calibrated artifact exists, no code change is needed
here either — `_select_columns()` re-reads REQUIRED_FEATURES at query
time, so this file has no hard-coded feature list to update.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import bindparam, text

from app.ml.explainibility_inference.inference.feature_contract import ID_COLUMN, REQUIRED_FEATURES

TABLE_NAME = "features_encoded"


class CustomerNotFoundError(LookupError):
    """Raised when none of the requested customer_unique_id(s) exist in the customer_features table."""


def _get_engine():
    # Imported lazily (inside the function, not at module import time) so
    # that importing this module — or anything that imports it, like the
    # FastAPI app at startup — does NOT require DB_* env vars to be set
    # unless a query is actually run. This keeps predict()/SHAP/reason-code
    # unit tests fully DB-free, matching the existing test suite's setup.
    from app.Database.database import engine

    return engine


def _select_columns() -> list[str]:
    # Single source of truth stays in feature_contract.py — this file
    # never re-lists feature names itself.
    return [ID_COLUMN] + REQUIRED_FEATURES


def get_customer_features(customer_unique_id: str) -> pd.DataFrame:
    """
    Fetch one customer's feature row from `customer_features`.

    Returns a 1-row DataFrame shaped exactly like feature_contract expects.
    Raises CustomerNotFoundError if the id doesn't exist in the table.
    """
    return get_customer_features_batch([customer_unique_id])


def get_customer_features_batch(customer_unique_ids: list[str]) -> pd.DataFrame:
    """
    Fetch multiple customers' feature rows in a single read-only query.

    Ids that don't exist are simply absent from the result — UNLESS none
    of the requested ids matched anything, in which case
    CustomerNotFoundError is raised (an empty DataFrame reaching
    validate_features() would otherwise fail with a less useful "input is
    empty" error).
    """
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
    """
    Fetch every row of `features_encoded` — used by the batch job
    (generate_predictions_table.py) that scores the whole customer base
    and writes churn_predictions for the segmentation module, as opposed
    to the on-demand single/batch lookups above (used by the API).

    Returns a DataFrame shaped exactly like the other getters (ID_COLUMN +
    REQUIRED_FEATURES), ready to pass straight into ChurnPredictor.predict().
    Raises CustomerNotFoundError if the table is empty.
    """
    columns = _select_columns()
    column_list_sql = ", ".join(columns)

    query = text(f"SELECT {column_list_sql} FROM {TABLE_NAME}")  # nosec - table/column names are code

    engine = _get_engine()
    with engine.connect() as connection:
        result = pd.read_sql(query, connection)

    if result.empty:
        raise CustomerNotFoundError(f"`{TABLE_NAME}` has no rows to score.")

    return result[columns].reset_index(drop=True)
