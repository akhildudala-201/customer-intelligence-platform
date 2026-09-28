from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import APIRouter, HTTPException, Query

from app.Database.database import engine

from app.api.segmentation_and_risk.schemas.segmentation_schemas import (
    AtRiskResponse,
    CustomerProfileResponse,
    CustomerSearchResponse,
    CustomersBySegmentResponse,
    RiskSummaryResponse,
    RiskThresholdsResponse,
    SegmentCustomersResponse,
    SegmentProfileResponse,
    SegmentRiskMixResponse,
    SegmentSummaryResponse,
    SegmentValueMixResponse,
)


router = APIRouter(
    tags=["Segmentation & Risk"]
)



CUSTOMER_INTELLIGENCE_TABLE = "customer_intelligence_base"
RISK_TABLE = "customer_risk_tiers"
SEGMENTS_TABLE = "customer_segments"
CLV_TABLE = "customer_clv"


# COLUMN NAMES

ID_COLUMN = "customer_unique_id"
CHURN_COLUMN = "churn_probability"
RISK_COLUMN = "risk_tier"
SEGMENT_ID_COLUMN = "segment_id"
SEGMENT_LABEL_COLUMN = "segment_label"
VALUE_TIER_COLUMN = "value_tier"
CLV_COLUMN = "clv"


def read_table(table_name: str) -> pd.DataFrame:
    """
    Read a table from the database into a pandas DataFrame.
    """

    try:
        return pd.read_sql_table(
            table_name,
            con=engine,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to read database table "
                f"'{table_name}': {str(exc)}"
            ),
        )


def clean_value(value: Any) -> Any:
    """
    Convert pandas/NumPy values into JSON-safe Python values.
    """

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def dataframe_to_records(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Convert DataFrame rows into JSON-safe dictionaries.
    """

    if df.empty:
        return []

    records = df.to_dict(
        orient="records"
    )

    cleaned_records = []

    for record in records:
        cleaned_record = {
            key: clean_value(value)
            for key, value in record.items()
        }

        cleaned_records.append(
            cleaned_record
        )

    return cleaned_records


def require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """
    Validate that required columns exist in a DataFrame.
    """

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Table `{table_name}` is missing required "
                f"column(s): {missing}"
            ),
        )



def load_customer_base() -> pd.DataFrame:
    """
    Load the main customer intelligence table.
    """

    df = read_table(
        CUSTOMER_INTELLIGENCE_TABLE
    )

    require_columns(
        df,
        [
            ID_COLUMN,
            CHURN_COLUMN,
        ],
        CUSTOMER_INTELLIGENCE_TABLE,
    )

    return df


def load_segments() -> pd.DataFrame:
    """
    Load customer segmentation information.
    """

    df = read_table(
        SEGMENTS_TABLE
    )

    require_columns(
        df,
        [
            ID_COLUMN,
            SEGMENT_ID_COLUMN,
        ],
        SEGMENTS_TABLE,
    )

    return df

def load_risk() -> pd.DataFrame:
    """
    Load customer risk-tier information.
    """

    df = read_table(
        RISK_TABLE
    )

    require_columns(
        df,
        [
            ID_COLUMN,
            RISK_COLUMN,
        ],
        RISK_TABLE,
    )

    return df



def load_clv_if_available() -> pd.DataFrame | None:
    """
    Load CLV data if the CLV table exists.

    If the table cannot be loaded, return None so that
    APIs that do not require CLV can continue working.
    """

    try:
        df = read_table(
            CLV_TABLE
        )

        if ID_COLUMN not in df.columns:
            return None

        return df

    except HTTPException:
        return None



def build_customer_view() -> pd.DataFrame:
    """
    Build a combined customer-level DataFrame by merging:

    - customer intelligence
    - customer segmentation
    - customer risk
    - customer CLV/value information
    """

    customer_df = load_customer_base()


    try:
        segment_df = load_segments()

        segment_columns = [
            ID_COLUMN,
            SEGMENT_ID_COLUMN,
        ]

        if SEGMENT_LABEL_COLUMN in segment_df.columns:
            segment_columns.append(
                SEGMENT_LABEL_COLUMN
            )

        segment_df = segment_df[
            segment_columns
        ].copy()

        segment_df = segment_df.drop_duplicates(
            subset=[ID_COLUMN]
        )

        customer_df = customer_df.merge(
            segment_df,
            on=ID_COLUMN,
            how="left",
            suffixes=("", "_segment"),
        )

        # Handle duplicate segment columns

        if (
            f"{SEGMENT_ID_COLUMN}_segment"
            in customer_df.columns
        ):
            if SEGMENT_ID_COLUMN not in customer_df.columns:
                customer_df[
                    SEGMENT_ID_COLUMN
                ] = customer_df[
                    f"{SEGMENT_ID_COLUMN}_segment"
                ]

            customer_df.drop(
                columns=[
                    f"{SEGMENT_ID_COLUMN}_segment"
                ],
                inplace=True,
            )

        if (
            f"{SEGMENT_LABEL_COLUMN}_segment"
            in customer_df.columns
        ):
            if SEGMENT_LABEL_COLUMN not in customer_df.columns:
                customer_df[
                    SEGMENT_LABEL_COLUMN
                ] = customer_df[
                    f"{SEGMENT_LABEL_COLUMN}_segment"
                ]

            customer_df.drop(
                columns=[
                    f"{SEGMENT_LABEL_COLUMN}_segment"
                ],
                inplace=True,
            )

    except HTTPException:
        pass

  
    try:
        risk_df = load_risk()

        risk_columns = [
            ID_COLUMN,
            RISK_COLUMN,
        ]

        if "cluster_probability" in risk_df.columns:
            risk_columns.append(
                "cluster_probability"
            )

        risk_df = risk_df[
            risk_columns
        ].copy()

        risk_df = risk_df.drop_duplicates(
            subset=[ID_COLUMN]
        )

        # If risk_tier already exists in customer_df,
        # merge only the missing risk information.

        if RISK_COLUMN in customer_df.columns:
            risk_df = risk_df[
                [
                    column
                    for column in risk_df.columns
                    if column != RISK_COLUMN
                ]
            ]

        customer_df = customer_df.merge(
            risk_df,
            on=ID_COLUMN,
            how="left",
            suffixes=("", "_risk"),
        )

        # Resolve duplicate cluster probability

        if (
            "cluster_probability_risk"
            in customer_df.columns
        ):
            if "cluster_probability" not in customer_df.columns:
                customer_df[
                    "cluster_probability"
                ] = customer_df[
                    "cluster_probability_risk"
                ]

            customer_df.drop(
                columns=[
                    "cluster_probability_risk"
                ],
                inplace=True,
            )

    except HTTPException:
        pass


    clv_df = load_clv_if_available()

    if clv_df is not None:

        clv_columns = [
            ID_COLUMN
        ]

        if CLV_COLUMN in clv_df.columns:
            clv_columns.append(
                CLV_COLUMN
            )

        if VALUE_TIER_COLUMN in clv_df.columns:
            clv_columns.append(
                VALUE_TIER_COLUMN
            )

        if len(clv_columns) > 1:

            clv_df = clv_df[
                clv_columns
            ].copy()

            clv_df = clv_df.drop_duplicates(
                subset=[ID_COLUMN]
            )

            # Avoid duplicate columns if CLV/value tier
            # already exists in the customer table.

            columns_to_merge = [
                ID_COLUMN
            ]

            for column in clv_columns:
                if column == ID_COLUMN:
                    continue

                if column not in customer_df.columns:
                    columns_to_merge.append(
                        column
                    )

            clv_df = clv_df[
                columns_to_merge
            ]

            customer_df = customer_df.merge(
                clv_df,
                on=ID_COLUMN,
                how="left",
            )


    if CHURN_COLUMN in customer_df.columns:
        customer_df[CHURN_COLUMN] = pd.to_numeric(
            customer_df[CHURN_COLUMN],
            errors="coerce",
        )

    if SEGMENT_ID_COLUMN in customer_df.columns:
        customer_df[SEGMENT_ID_COLUMN] = pd.to_numeric(
            customer_df[SEGMENT_ID_COLUMN],
            errors="coerce",
        )

    if CLV_COLUMN in customer_df.columns:
        customer_df[CLV_COLUMN] = pd.to_numeric(
            customer_df[CLV_COLUMN],
            errors="coerce",
        )

    if "cluster_probability" in customer_df.columns:
        customer_df["cluster_probability"] = pd.to_numeric(
            customer_df["cluster_probability"],
            errors="coerce",
        )

    return customer_df


@router.get(
    "/customers/bysegment",
    response_model=CustomersBySegmentResponse,
    summary="Get customers by segment",
)
def get_customers_by_segment(
    segment_id: int | None = None,
    segment_label: str | None = None,
    risk_tier: str | None = None,
    value_tier: str | None = None,
    limit: int = Query(
        50,
        ge=1,
        le=500,
    ),
    offset: int = Query(
        0,
        ge=0,
    ),
):
    """
    Return customers filtered by segment,
    risk tier, or value tier.
    """

    df = build_customer_view()

    # Segment ID filter

    if segment_id is not None:
        df = df[
            pd.to_numeric(
                df[SEGMENT_ID_COLUMN],
                errors="coerce",
            )
            == segment_id
        ]

    # Segment label filter

    if segment_label is not None:
        if SEGMENT_LABEL_COLUMN in df.columns:
            df = df[
                df[SEGMENT_LABEL_COLUMN]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                == segment_label.strip().lower()
            ]

    # Risk filter

    if risk_tier is not None:
        if RISK_COLUMN in df.columns:
            df = df[
                df[RISK_COLUMN]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                == risk_tier.strip().lower()
            ]

    # Value tier filter

    if value_tier is not None:
        if VALUE_TIER_COLUMN in df.columns:
            df = df[
                df[VALUE_TIER_COLUMN]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                == value_tier.strip().lower()
            ]

    total = len(df)

    page_df = df.iloc[
        offset: offset + limit
    ]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "customers": dataframe_to_records(
            page_df
        ),
    }



@router.get(
    "/segments/profile",
    response_model=SegmentProfileResponse,
    summary="Get segment profiles",
)
def get_segment_profiles():
    """
    Return customer count, average churn probability,
    and average CLV for each segment.
    """

    df = build_customer_view()

    require_columns(
        df,
        [
            SEGMENT_ID_COLUMN,
        ],
        "customer intelligence view",
    )

    group_columns = [
        SEGMENT_ID_COLUMN
    ]

    if SEGMENT_LABEL_COLUMN in df.columns:
        group_columns.append(
            SEGMENT_LABEL_COLUMN
        )

    grouped = (
        df.groupby(
            group_columns,
            dropna=False,
        )
        .agg(
            customer_count=(
                ID_COLUMN,
                "count",
            ),
            average_churn_probability=(
                CHURN_COLUMN,
                "mean",
            ),
        )
        .reset_index()
    )

    if CLV_COLUMN in df.columns:
        clv_group = (
            df.groupby(
                group_columns,
                dropna=False,
            )[CLV_COLUMN]
            .mean()
            .reset_index(
                name="average_clv"
            )
        )

        grouped = grouped.merge(
            clv_group,
            on=group_columns,
            how="left",
        )

    else:
        grouped["average_clv"] = None

    return {
        "segments": dataframe_to_records(
            grouped
        )
    }


@router.get(
    "/segments/{segment_id}/customers",
    response_model=SegmentCustomersResponse,
    summary="Get customers inside a segment",
)
def get_segment_customers(
    segment_id: int,
    risk_tier: str | None = None,
    limit: int = Query(
        50,
        ge=1,
        le=500,
    ),
    offset: int = Query(
        0,
        ge=0,
    ),
):
    """
    Return customers belonging to a specific segment.
    """

    df = build_customer_view()

    df = df[
        pd.to_numeric(
            df[SEGMENT_ID_COLUMN],
            errors="coerce",
        )
        == segment_id
    ]

    if risk_tier is not None:
        df = df[
            df[RISK_COLUMN]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            == risk_tier.strip().lower()
        ]

    total = len(df)

    page_df = df.iloc[
        offset: offset + limit
    ]

    return {
        "segment_id": segment_id,
        "total": total,
        "limit": limit,
        "offset": offset,
        "customers": dataframe_to_records(
            page_df
        ),
    }



@router.get(
    "/segments/risk-mix",
    response_model=SegmentRiskMixResponse,
    summary="Get risk distribution inside segments",
)
def get_segment_risk_mix():
    """
    Return the number of customers in each
    risk tier for each segment.
    """

    df = build_customer_view()

    require_columns(
        df,
        [
            SEGMENT_ID_COLUMN,
            RISK_COLUMN,
        ],
        "customer intelligence view",
    )

    group_columns = [
        SEGMENT_ID_COLUMN
    ]

    if SEGMENT_LABEL_COLUMN in df.columns:
        group_columns.append(
            SEGMENT_LABEL_COLUMN
        )

    group_columns.append(
        RISK_COLUMN
    )

    grouped = (
        df.groupby(
            group_columns,
            dropna=False,
        )
        .size()
        .reset_index(
            name="customer_count"
        )
    )

    return {
        "risk_mix": dataframe_to_records(
            grouped
        )
    }



@router.get(
    "/risk/summary",
    response_model=RiskSummaryResponse,
    summary="Get overall customer risk summary",
)
def get_risk_summary():
    """
    Return total customers, average churn probability,
    and risk-tier distribution.
    """

    df = build_customer_view()

    require_columns(
        df,
        [
            CHURN_COLUMN,
            RISK_COLUMN,
        ],
        "customer intelligence view",
    )

    total_customers = len(df)

    average_churn_probability = (
        float(
            df[CHURN_COLUMN].mean()
        )
        if total_customers
        else 0.0
    )

    risk_counts = (
        df[RISK_COLUMN]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = [
        RISK_COLUMN,
        "customer_count",
    ]

    if total_customers > 0:
        risk_counts["percentage"] = (
            risk_counts["customer_count"]
            / total_customers
            * 100
        )
    else:
        risk_counts["percentage"] = 0.0

    return {
        "total_customers": total_customers,
        "average_churn_probability": (
            average_churn_probability
        ),
        "risk_distribution": dataframe_to_records(
            risk_counts
        ),
    }



@router.get(
    "/atrisk",
    response_model=AtRiskResponse,
    summary="Get at-risk customers",
)
def get_at_risk_customers(
    risk_tier: str | None = None,
    min_probability: float | None = Query(
        None,
        ge=0,
        le=1,
    ),
    max_probability: float | None = Query(
        None,
        ge=0,
        le=1,
    ),
    segment_id: int | None = None,
    segment_label: str | None = None,
    value_tier: str | None = None,
    limit: int = Query(
        50,
        ge=1,
        le=500,
    ),
    offset: int = Query(
        0,
        ge=0,
    ),
):
    """
    Return customers considered at risk.

    Supports filtering by:
    - risk tier
    - churn probability
    - segment
    - value tier
    """

    if (
        min_probability is not None
        and max_probability is not None
        and min_probability > max_probability
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "min_probability cannot be greater "
                "than max_probability."
            ),
        )

    df = build_customer_view()

    require_columns(
        df,
        [
            CHURN_COLUMN,
        ],
        "customer intelligence view",
    )

    # Risk tier

    if risk_tier is not None:
        if RISK_COLUMN in df.columns:
            df = df[
                df[RISK_COLUMN]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                == risk_tier.strip().lower()
            ]

    # Minimum churn probability

    if min_probability is not None:
        df = df[
            pd.to_numeric(
                df[CHURN_COLUMN],
                errors="coerce",
            )
            >= min_probability
        ]

    # Maximum churn probability

    if max_probability is not None:
        df = df[
            pd.to_numeric(
                df[CHURN_COLUMN],
                errors="coerce",
            )
            <= max_probability
        ]

    # Segment ID

    if segment_id is not None:
        df = df[
            pd.to_numeric(
                df[SEGMENT_ID_COLUMN],
                errors="coerce",
            )
            == segment_id
        ]

    # Segment label

    if segment_label is not None:
        if SEGMENT_LABEL_COLUMN in df.columns:
            df = df[
                df[SEGMENT_LABEL_COLUMN]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                == segment_label.strip().lower()
            ]

    # Value tier

    if value_tier is not None:
        if VALUE_TIER_COLUMN in df.columns:
            df = df[
                df[VALUE_TIER_COLUMN]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                == value_tier.strip().lower()
            ]

    # Highest churn probability first

    df = df.sort_values(
        by=CHURN_COLUMN,
        ascending=False,
        na_position="last",
    )

    total = len(df)

    page_df = df.iloc[
        offset: offset + limit
    ]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "customers": dataframe_to_records(
            page_df
        ),
    }


@router.get(
    "/customers/{customer_id}/profile",
    response_model=CustomerProfileResponse,
    summary="Get customer profile",
)
def get_customer_profile(
    customer_id: str,
):
    """
    Return the complete profile of one customer.
    """

    df = build_customer_view()

    require_columns(
        df,
        [
            ID_COLUMN,
        ],
        "customer intelligence view",
    )

    customer = df[
        df[ID_COLUMN]
        .astype(str)
        == str(customer_id)
    ]

    if customer.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Customer '{customer_id}' "
                f"was not found."
            ),
        )

    profile = dataframe_to_records(
        customer.iloc[[0]]
    )[0]

    return {
        "customer_id": customer_id,
        "profile": profile,
    }



@router.get(
    "/segments/summary",
    response_model=SegmentSummaryResponse,
    summary="Get segment summary",
)
def get_segment_summary():
    """
    Return overall statistics for each customer segment.
    """

    df = build_customer_view()

    require_columns(
        df,
        [
            ID_COLUMN,
            SEGMENT_ID_COLUMN,
            CHURN_COLUMN,
        ],
        "customer intelligence view",
    )

    total_customers = len(df)

    group_columns = [
        SEGMENT_ID_COLUMN
    ]

    if SEGMENT_LABEL_COLUMN in df.columns:
        group_columns.append(
            SEGMENT_LABEL_COLUMN
        )

    grouped = (
        df.groupby(
            group_columns,
            dropna=False,
        )
        .agg(
            customer_count=(
                ID_COLUMN,
                "count",
            ),
            average_churn_probability=(
                CHURN_COLUMN,
                "mean",
            ),
        )
        .reset_index()
    )

    if total_customers > 0:
        grouped["customer_percentage"] = (
            grouped["customer_count"]
            / total_customers
            * 100
        )
    else:
        grouped["customer_percentage"] = 0.0

    return {
        "total_customers": total_customers,
        "segment_count": len(grouped),
        "segments": dataframe_to_records(
            grouped
        ),
    }



@router.get(
    "/segments/value-mix",
    response_model=SegmentValueMixResponse,
    summary="Get value distribution inside segments",
)
def get_segment_value_mix():
    """
    Return customer value-tier distribution
    for each segment.
    """

    df = build_customer_view()

    if VALUE_TIER_COLUMN not in df.columns:
        raise HTTPException(
            status_code=500,
            detail=(
                "Value tier information is not "
                "available in the customer data."
            ),
        )

    require_columns(
        df,
        [
            SEGMENT_ID_COLUMN,
            VALUE_TIER_COLUMN,
        ],
        "customer intelligence view",
    )

    group_columns = [
        SEGMENT_ID_COLUMN
    ]

    if SEGMENT_LABEL_COLUMN in df.columns:
        group_columns.append(
            SEGMENT_LABEL_COLUMN
        )

    group_columns.append(
        VALUE_TIER_COLUMN
    )

    grouped = (
        df.groupby(
            group_columns,
            dropna=False,
        )
        .size()
        .reset_index(
            name="customer_count"
        )
    )

    return {
        "value_mix": dataframe_to_records(
            grouped
        )
    }


@router.get(
    "/risk/thresholds",
    response_model=RiskThresholdsResponse,
    summary="Get risk classification thresholds",
)
def get_risk_thresholds():
    """
    Return risk-tier thresholds from the YAML configuration.
    """

    current_file = Path(__file__).resolve()


    project_root = current_file.parents[3]

    candidate_paths = [
        # app/config/risk_tier_thresholds.yaml
        project_root
        / "app"
        / "config"
        / "risk_tier_thresholds.yaml",

    
        project_root
        / "app"
        / "config"
        / "risk_tier_thresholds.yml",


        project_root
        / "app"
        / "segmentation"
        / "config"
        / "risk_tier_thresholds.yaml",

       
        project_root
        / "app"
        / "segmentation"
        / "customer_intelligence"
        / "config"
        / "risk_tier_thresholds.yaml",

        # root config
        project_root
        / "config"
        / "risk_tier_thresholds.yaml",
    ]

  

    config_path = None

    for path in candidate_paths:
        if path.exists():
            config_path = path
            break

  
    if config_path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "risk_tier_thresholds.yaml "
                "could not be found."
            ),
        )

   

    try:
        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(file)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to read risk threshold "
                f"configuration: {str(exc)}"
            ),
        )



    tiers = config.get(
        "tiers",
        [],
    )

    thresholds = []

    for tier in tiers:
        thresholds.append(
            {
                "name": tier.get("name"),
                "upper_bound": float(
                    tier.get("upper_bound")
                ),
            }
        )

    return {
        "thresholds": thresholds,
        "source": str(config_path),
    }


@router.get(
    "/customers/search",
    response_model=CustomerSearchResponse,
    summary="Search customers",
)
def search_customers(
    q: str = Query(
        ...,
        min_length=1,
    ),
    risk_tier: str | None = None,
    segment_id: int | None = None,
    limit: int = Query(
        50,
        ge=1,
        le=500,
    ),
    offset: int = Query(
        0,
        ge=0,
    ),
):
    """
    Search customers using customer ID,
    segment label, risk tier, or value tier.
    """

    df = build_customer_view()



    if risk_tier is not None:

        if RISK_COLUMN in df.columns:

            df = df[
                df[RISK_COLUMN]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                == risk_tier.strip().lower()
            ]

  

    if segment_id is not None:

        if SEGMENT_ID_COLUMN in df.columns:

            df = df[
                pd.to_numeric(
                    df[SEGMENT_ID_COLUMN],
                    errors="coerce",
                )
                == segment_id
            ]

  

    searchable_columns = [
        ID_COLUMN,
        SEGMENT_LABEL_COLUMN,
        RISK_COLUMN,
        VALUE_TIER_COLUMN,
    ]

    searchable_columns = [
        column
        for column in searchable_columns
        if column in df.columns
    ]

    if not searchable_columns:
        raise HTTPException(
            status_code=404,
            detail=(
                "No searchable customer columns "
                "are available."
            ),
        )

 
    search_text = q.strip().lower()

    mask = pd.Series(
        False,
        index=df.index,
    )

    for column in searchable_columns:

        mask = (
            mask
            | df[column]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(
                search_text,
                regex=False,
                na=False,
            )
        )

    filtered_df = df[
        mask
    ].copy()

  

    total = len(filtered_df)

    page_df = filtered_df.iloc[
        offset: offset + limit
    ]

    customers = dataframe_to_records(
        page_df
    )

    return {
        "query": q,
        "total": total,
        "limit": limit,
        "offset": offset,
        "customers": customers,
    }
