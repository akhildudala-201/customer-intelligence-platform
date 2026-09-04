
"""
Pre-load data cleaning rules.

This file cleans each CSV DataFrame before it is written to MySQL.

Cleaning strategy:
1. Remove unnecessary whitespace from text columns.
2. Convert empty strings to missing values (NA).
3. Remove duplicate records.
4. Convert expected numeric columns to numeric values.
5. Remove rows with invalid negative values where negatives are impossible.
6. Validate allowed ranges, such as review scores and coordinates.
7. Remove rows where required columns are missing.
8. Remove rows whose foreign-key parent does not exist.
9. Remove rows only when MORE than 50% of their columns are missing.

Important:
- A missing REQUIRED value always removes the row.
- Missing OPTIONAL values are allowed.
- A row with exactly 50% missing values is kept.
- A row with MORE than 50% missing values is removed.
"""

import pandas as pd


# ============================================================
# REPORTING
# ============================================================

def _report(table_name, rule, before, after):
    """
    Print how many rows were removed by a cleaning rule.

    Example:
    customers duplicate rows: dropped 10 row(s)
    """

    dropped = before - after

    if dropped > 0:
        print(
            f"  [{table_name}] {rule}: "
            f"dropped {dropped} row(s)"
        )


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def _dedupe(df, table_name, subset=None):
    """
    Remove duplicate rows.

    subset:
        If provided, duplicates are determined using only
        the specified column(s).

    Example:
        subset=["customer_id"]

    means that customer_id must be unique.
    """

    before = len(df)

    df = df.drop_duplicates(subset=subset)

    _report(
        table_name,
        f"duplicate rows (subset={subset or 'all columns'})",
        before,
        len(df),
    )

    return df


# ============================================================
# NUMERIC CONVERSION
# ============================================================

def _coerce_numeric(df, columns):
    """
    Convert specified columns to numeric values.

    Invalid values are converted to NaN.

    Example:
        "100" -> 100
        "abc" -> NaN

    Later cleaning rules can remove rows where important
    numeric values became NaN.
    """

    for col in columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    return df


# ============================================================
# STRING CLEANING
# ============================================================

def _strip_strings(df):
    """
    Clean text columns.

    Actions:
    1. Remove leading and trailing whitespace.
    2. Convert empty strings into pd.NA.

    Example:

        " Hyderabad " -> "Hyderabad"

        "" -> NA
    """

    str_cols = df.select_dtypes(include="object").columns

    for col in str_cols:

        # Remove leading and trailing whitespace.
        df[col] = df[col].str.strip()

        # Treat empty strings as missing values.
        df[col] = df[col].replace("", pd.NA)

    return df


# ============================================================
# REQUIRED COLUMN VALIDATION
# ============================================================

def _drop_missing_required(df, table_name, required_columns):
    """
    Remove rows where ANY required column is missing.

    This rule is separate from the 50% NULL rule.

    Example:

    customer_id is required.

    Even if a row has only 10% missing values, the row
    is removed if customer_id is missing.

    This prevents invalid records from being loaded into MySQL.
    """

    # Only keep required columns that actually exist in the DataFrame.
    required_columns = [
        col
        for col in required_columns
        if col in df.columns
    ]

    # If none of the specified columns exist, do nothing.
    if not required_columns:
        return df

    before = len(df)

    # Drop a row if ANY required column is missing.
    df = df.dropna(subset=required_columns)

    _report(
        table_name,
        f"missing required column(s): {required_columns}",
        before,
        len(df),
    )

    return df


# ============================================================
# NEGATIVE VALUE VALIDATION
# ============================================================

def _drop_negative(df, table_name, columns):
    """
    Remove rows containing negative values in columns where
    negative values are physically or logically impossible.

    Examples:
        price < 0          -> invalid
        weight < 0         -> invalid
        freight_value < 0  -> invalid
    """

    before = len(df)

    for col in columns:

        if col in df.columns:

            # Keep rows where:
            # - value is missing, OR
            # - value is greater than or equal to zero.
            #
            # Missing values may later be handled by required-column
            # or NULL-percentage validation.
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


# ============================================================
# RANGE VALIDATION
# ============================================================

def _clamp_out_of_range(
    df,
    table_name,
    column,
    low,
    high,
):
    """
    Mark values outside the allowed range as missing.

    Example:

        review_score must be between 1 and 5.

        latitude must be between -90 and 90.

    Invalid values are converted to NA.
    Required-column validation or the NULL-percentage rule
    will then decide whether the row should be removed.
    """

    if column not in df.columns:
        return df

    before_valid = df[column].notna().sum()

    # Find values below the minimum or above the maximum.
    mask = (
        (df[column] < low)
        | (df[column] > high)
    )

    # Convert invalid values to missing.
    df.loc[mask, column] = pd.NA

    after_valid = df[column].notna().sum()

    _report(
        table_name,
        f"{column} outside [{low}, {high}]",
        before_valid,
        after_valid,
    )

    return df


# ============================================================
# NULL PERCENTAGE VALIDATION
# ============================================================

def _drop_high_null_rows(
    df,
    table_name,
    threshold=0.5,
):
    """
    Remove rows only when MORE than the specified percentage
    of columns are missing.

    threshold=0.5 means:

        0% missing   -> KEEP
        25% missing  -> KEEP
        50% missing  -> KEEP
        51% missing  -> DROP
        80% missing  -> DROP

    Required columns are handled separately by
    _drop_missing_required().
    """

    before = len(df)

    # Calculate missing-value percentage for every row.
    #
    # axis=1 means calculate across each row.
    #
    # Example:
    #
    # 2 missing columns out of 5 columns
    #
    # 2 / 5 = 0.4 = 40%
    null_percentage = df.isna().mean(axis=1)

    # Keep rows with missing values less than or equal to
    # the threshold.
    #
    # <= 0.5 means exactly 50% missing is kept.
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


# ============================================================
# FOREIGN KEY VALIDATION
# ============================================================

def _drop_orphan_fk(
    df,
    table_name,
    column,
    valid_keys,
    parent_name,
):
    """
    Remove rows whose foreign key does not exist in the
    surviving parent table.

    Example:

        orders.customer_id
            references
        customers.customer_id

    If customer C001 was removed during customer cleaning,
    an order pointing to C001 should also be removed.

    Otherwise MySQL may reject the row because of the
    foreign key constraint.
    """

    # Skip validation if:
    # - valid_keys is not available,
    # - the parent table is not available,
    # - the foreign-key column does not exist.
    if (
        valid_keys is None
        or parent_name not in valid_keys
        or column not in df.columns
    ):
        return df

    before = len(df)

    # Keep only rows whose foreign key exists in the
    # parent's surviving key set.
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


# ============================================================
# MAIN CLEANING FUNCTION
# ============================================================

def clean_dataframe(
    table_name,
    df,
    valid_keys=None,
):
    """
    Main entry point for cleaning a DataFrame.

    Each table has its own cleaning rules.

    Overall process:

        Raw DataFrame
            ↓
        Clean text values
            ↓
        Table-specific cleaning
            ↓
        Required-column validation
            ↓
        Numeric/range validation
            ↓
        Foreign-key validation
            ↓
        Drop rows with MORE than 50% missing values
            ↓
        Return cleaned DataFrame
    """

    # --------------------------------------------------------
    # STEP 1: Clean text columns for every table.
    # --------------------------------------------------------

    df = _strip_strings(df)

    # ========================================================
    # CUSTOMERS
    # ========================================================

    if table_name == "customers":

        # customer_id should uniquely identify a customer.
        df = _dedupe(
            df,
            table_name,
            subset=["customer_id"],
        )

        # A customer without an ID is not usable.
        df = _drop_missing_required(
            df,
            table_name,
            [
                "customer_id",
            ],
        )

    # ========================================================
    # SELLERS
    # ========================================================

    elif table_name == "sellers":

        # seller_id should be unique.
        df = _dedupe(
            df,
            table_name,
            subset=["seller_id"],
        )

        # A seller without an ID is not usable.
        df = _drop_missing_required(
            df,
            table_name,
            [
                "seller_id",
            ],
        )

    # ========================================================
    # PRODUCTS
    # ========================================================

    elif table_name == "products":

        # product_id should be unique.
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

        # Convert product measurements to numeric values.
        df = _coerce_numeric(
            df,
            numeric_cols,
        )

        # A product must have a product ID.
        #
        # Measurements can be missing, and the 50% NULL rule
        # will handle rows with too much missing information.
        df = _drop_missing_required(
            df,
            table_name,
            [
                "product_id",
            ],
        )

        # Product dimensions and weight cannot be negative.
        df = _drop_negative(
            df,
            table_name,
            numeric_cols,
        )

    # ========================================================
    # ORDERS
    # ========================================================

    elif table_name == "orders":

        # order_id should be unique.
        df = _dedupe(
            df,
            table_name,
            subset=["order_id"],
        )

        # An order must point to an existing customer.
        df = _drop_orphan_fk(
            df,
            table_name,
            "customer_id",
            valid_keys,
            "customers",
        )

        # Required fields for an order.
        #
        # Delivery-related timestamps are NOT included because
        # they can legitimately be NULL for orders that are:
        #
        # canceled
        # processing
        # invoiced
        # shipped
        #
        # etc.
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

        # IMPORTANT:
        # Do NOT return here.
        #
        # The order must continue to the final
        # _drop_high_null_rows() check below.

    # ========================================================
    # ORDER ITEMS
    # ========================================================

    elif table_name == "order_items":

        # Convert numeric values first.
        #
        # Example:
        # "abc" -> NaN
        #
        # Then required-column validation can remove the row.
        df = _coerce_numeric(
            df,
            [
                "price",
                "freight_value",
            ],
        )

        # Required fields.
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

        # Price and freight cannot be negative.
        df = _drop_negative(
            df,
            table_name,
            [
                "price",
                "freight_value",
            ],
        )

        # Validate parent relationships.
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

    # ========================================================
    # ORDER PAYMENTS
    # ========================================================

    elif table_name == "order_payments":

        # Convert payment value to numeric.
        df = _coerce_numeric(
            df,
            [
                "payment_value",
            ],
        )

        # Required fields.
        df = _drop_missing_required(
            df,
            table_name,
            [
                "order_id",
                "payment_value",
            ],
        )

        # Payment cannot be negative.
        df = _drop_negative(
            df,
            table_name,
            [
                "payment_value",
            ],
        )

        # The referenced order must exist.
        df = _drop_orphan_fk(
            df,
            table_name,
            "order_id",
            valid_keys,
            "orders",
        )

    # ========================================================
    # ORDER REVIEWS
    # ========================================================

    elif table_name == "order_reviews":

        # Convert review score to numeric.
        df = _coerce_numeric(
            df,
            [
                "review_score",
            ],
        )

        # Review score must be between 1 and 5.
        #
        # Invalid values become NA.
        df = _clamp_out_of_range(
            df,
            table_name,
            "review_score",
            1,
            5,
        )

        # Required review fields.
        #
        # Review title and review message are NOT required.
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

        # The referenced order must exist.
        df = _drop_orphan_fk(
            df,
            table_name,
            "order_id",
            valid_keys,
            "orders",
        )

        # Written comments are optional.
        #
        # A customer can give only a star rating without
        # writing a title or message.
        #
        # Therefore, keep the review and replace missing
        # comments with empty strings.
        for col in (
            "review_comment_title",
            "review_comment_message",
        ):

            if col in df.columns:

                df[col] = df[col].fillna("")

    # ========================================================
    # GEOLOCATION
    # ========================================================

    elif table_name == "geolocation":

        # Convert latitude and longitude to numeric.
        df = _coerce_numeric(
            df,
            [
                "geolocation_lat",
                "geolocation_lng",
            ],
        )

        # Required geolocation fields.
        df = _drop_missing_required(
            df,
            table_name,
            [
                "geolocation_zip_code_prefix",
                "geolocation_lat",
                "geolocation_lng",
            ],
        )

        # Latitude must be between -90 and 90.
        df = _clamp_out_of_range(
            df,
            table_name,
            "geolocation_lat",
            -90,
            90,
        )

        # Longitude must be between -180 and 180.
        df = _clamp_out_of_range(
            df,
            table_name,
            "geolocation_lng",
            -180,
            180,
        )

        # Since invalid coordinates were converted to NA,
        # validate required fields again.
        df = _drop_missing_required(
            df,
            table_name,
            [
                "geolocation_zip_code_prefix",
                "geolocation_lat",
                "geolocation_lng",
            ],
        )

    # ========================================================
    # FINAL NULL-PERCENTAGE CHECK
    # ========================================================

    # Keep rows with:
    #
    # <= 50% missing values
    #
    # Remove rows with:
    #
    # > 50% missing values
    #
    # Required fields were already validated above.
    df = _drop_high_null_rows(
        df,
        table_name,
        threshold=0.5,
    )

    return df

