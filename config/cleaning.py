import pandas as pd
def _report(table_name, rule, before, after):
    # Print how many rows a cleaning rule dropped
    dropped = before - after
    if dropped > 0:
        print(
            f"  [{table_name}] {rule}: "
            f"dropped {dropped} row(s)"
        )

def _dedupe(df, table_name, subset=None):
    # Remove duplicate rows, optionally based on a subset of columns
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    _report(
        table_name,
        f"duplicate rows (subset={subset or 'all columns'})",
        before,
        len(df),
    )
    return df

def _coerce_numeric(df, columns):
    # Convert columns to numeric, turning invalid values into NaN
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )
    return df

def _strip_strings(df):
    # Trim whitespace in text columns and treat empty strings as missing
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace("", pd.NA)
    return df

def _drop_missing_required(df, table_name, required_columns):
    # Drop rows where any required column is missing
    required_columns = [
        col
        for col in required_columns
        if col in df.columns
    ]
    if not required_columns:
        return df
    before = len(df)
    df = df.dropna(subset=required_columns)
    _report(
        table_name,
        f"missing required column(s): {required_columns}",
        before,
        len(df),
    )
    return df

def _drop_negative(df, table_name, columns):
    # Drop rows with negative values in columns where negatives are impossible
    before = len(df)
    for col in columns:
        if col in df.columns:
            # Missing values pass through; required/null-% checks handle those
            df = df[
                df[col].isna() | (df[col] >= 0)
            ]
    _report(
        table_name,
        f"negative value in {columns}",
        before,
        len(df),
    )
    return df

def _clamp_out_of_range(
    df,
    table_name,
    column,
    low,
    high,
):
    # Convert out-of-range values to NA instead of dropping the row outright
    if column not in df.columns:
        return df
    before_valid = df[column].notna().sum()
    mask = (
        (df[column] < low)
        | (df[column] > high)
    )
    df.loc[mask, column] = pd.NA
    after_valid = df[column].notna().sum()
    _report(
        table_name,
        f"{column} outside [{low}, {high}]",
        before_valid,
        after_valid,
    )
    return df

def _drop_high_null_rows(
    df,
    table_name,
    threshold=0.5,
):
    # Drop rows with more than `threshold` fraction of columns missing
    before = len(df)
    null_percentage = df.isna().mean(axis=1)
    # <= threshold is kept, so exactly 50% missing survives
    df = df[
        null_percentage <= threshold
    ]
    _report(
        table_name,
        f"more than {threshold * 100}% missing values",
        before,
        len(df),
    )
    return df

def _drop_orphan_fk(
    df,
    table_name,
    column,
    valid_keys,
    parent_name,
):
    # Drop rows whose foreign key isn't in the parent table's surviving keys
    if (
        valid_keys is None
        or parent_name not in valid_keys
        or column not in df.columns
    ):
        return df
    before = len(df)
    df = df[
        df[column].isin(valid_keys[parent_name])
    ]
    _report(
        table_name,
        f"{column} not found in {parent_name} "
        f"(orphaned by upstream cleaning)",
        before,
        len(df),
    )
    return df

def clean_dataframe(
    table_name,
    df,
    valid_keys=None,
):
    # Main entry point: text cleanup -> per-table rules -> final null-% check
    df = _strip_strings(df)
    if table_name == "customers":
        df = _dedupe(
            df,
            table_name,
            subset=["customer_id"],
        )
        df = _drop_missing_required(
            df,
            table_name,
            [
                "customer_id",
            ],
        )
    elif table_name == "sellers":
        df = _dedupe(
            df,
            table_name,
            subset=["seller_id"],
        )
        df = _drop_missing_required(
            df,
            table_name,
            [
                "seller_id",
            ],
        )
    elif table_name == "products":
        df = _dedupe(
            df,
            table_name,
            subset=["product_id"],
        )
        numeric_cols = [
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ]
        df = _coerce_numeric(
            df,
            numeric_cols,
        )
        df = _drop_missing_required(
            df,
            table_name,
            [
                "product_id",
            ],
        )
        df = _drop_negative(
            df,
            table_name,
            numeric_cols,
        )
    elif table_name == "orders":
        df = _dedupe(
            df,
            table_name,
            subset=["order_id"],
        )
        df = _drop_orphan_fk(
            df,
            table_name,
            "customer_id",
            valid_keys,
            "customers",
        )
        # Delivery timestamps are excluded: legitimately NULL for
        # canceled/processing/invoiced/shipped orders
        df = _drop_missing_required(
            df,
            table_name,
            [
                "order_id",
                "customer_id",
                "order_status",
                "order_purchase_timestamp",
            ],
        )
    elif table_name == "order_items":
        df = _coerce_numeric(
            df,
            [
                "price",
                "freight_value",
            ],
        )
        df = _drop_missing_required(
            df,
            table_name,
            [
                "order_id",
                "product_id",
                "seller_id",
                "price",
                "freight_value",
            ],
        )
        df = _drop_negative(
            df,
            table_name,
            [
                "price",
                "freight_value",
            ],
        )
        df = _drop_orphan_fk(
            df,
            table_name,
            "order_id",
            valid_keys,
            "orders",
        )
        df = _drop_orphan_fk(
            df,
            table_name,
            "product_id",
            valid_keys,
            "products",
        )
        df = _drop_orphan_fk(
            df,
            table_name,
            "seller_id",
            valid_keys,
            "sellers",
        )
    elif table_name == "order_payments":
        df = _coerce_numeric(
            df,
            [
                "payment_value",
            ],
        )
        df = _drop_missing_required(
            df,
            table_name,
            [
                "order_id",
                "payment_value",
            ],
        )
        df = _drop_negative(
            df,
            table_name,
            [
                "payment_value",
            ],
        )
        df = _drop_orphan_fk(
            df,
            table_name,
            "order_id",
            valid_keys,
            "orders",
        )
    elif table_name == "order_reviews":
        df = _coerce_numeric(
            df,
            [
                "review_score",
            ],
        )
        df = _clamp_out_of_range(
            df,
            table_name,
            "review_score",
            1,
            5,
        )
        # Title and message are optional: a rating alone is valid
        df = _drop_missing_required(
            df,
            table_name,
            [
                "review_id",
                "order_id",
                "review_score",
                "review_creation_date",
            ],
        )
        df = _drop_orphan_fk(
            df,
            table_name,
            "order_id",
            valid_keys,
            "orders",
        )
        for col in (
            "review_comment_title",
            "review_comment_message",
        ):
            if col in df.columns:
                df[col] = df[col].fillna("")
    elif table_name == "geolocation":
        df = _coerce_numeric(
            df,
            [
                "geolocation_lat",
                "geolocation_lng",
            ],
        )
        df = _drop_missing_required(
            df,
            table_name,
            [
                "geolocation_zip_code_prefix",
                "geolocation_lat",
                "geolocation_lng",
            ],
        )
        df = _clamp_out_of_range(
            df,
            table_name,
            "geolocation_lat",
            -90,
            90,
        )
        df = _clamp_out_of_range(
            df,
            table_name,
            "geolocation_lng",
            -180,
            180,
        )
        # Re-check required fields since clamping may have introduced new NAs
        df = _drop_missing_required(
            df,
            table_name,
            [
                "geolocation_zip_code_prefix",
                "geolocation_lat",
                "geolocation_lng",
            ],
        )
    df = _drop_high_null_rows(
        df,
        table_name,
        threshold=0.5,
    )
    return df
