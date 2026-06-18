# =========================================================
# feature_engineering.py
# FINAL ENTERPRISE FEATURE ENGINEERING
# SAFE + TARGET PROTECTED + LOW NOISE
# =========================================================

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# IMPORTS
# =========================================================

import numpy as np
import pandas as pd

# =========================================================
# MODEL UTILS
# =========================================================

from models.model_utils import (
    detect_enterprise_targets
)

# =========================================================
# SAFE NUMERIC FEATURES
# =========================================================

def get_safe_numeric_features(
    df,
    protected_targets
):

    """
    Get safe numeric features only.
    """

    numeric_columns = df.select_dtypes(

        include=np.number
    ).columns.tolist()

    safe_columns = []

    excluded_keywords = [

        "_lag",

        "_rolling",

        "_trend",

        "_delta",

        "_interaction",

        "_ratio",

        "_poly",

        "_sin",

        "_cos",

        "id",

        "index"
    ]

    for column in numeric_columns:

        # =============================================
        # NEVER TOUCH TARGETS
        # =============================================

        if column in protected_targets:

            continue

        # =============================================
        # REMOVE ENGINEERED NOISE
        # =============================================

        if any(

            keyword in column.lower()

            for keyword in excluded_keywords
        ):

            continue

        safe_columns.append(column)

    return safe_columns


# =========================================================
# CREATE LAG FEATURES
# =========================================================

def create_lag_features(
    df,
    safe_features
):

    """
    Create enterprise lag features.
    """

    lag_features = []

    for column in safe_features:

        try:

            lag_column = f"{column}_lag_1"

            df[lag_column] = (
                df[column]
                .shift(1)
            )

            lag_features.append(
                lag_column
            )

        except:
            continue

    return df, lag_features


# =========================================================
# CREATE ROLLING FEATURES
# =========================================================

def create_rolling_features(
    df,
    safe_features
):

    """
    Create rolling mean features.
    """

    rolling_features = []

    for column in safe_features:

        try:

            rolling_column = (
                f"{column}_rolling_mean"
            )

            df[rolling_column] = (

                df[column]

                .rolling(window=3)

                .mean()
            )

            rolling_features.append(
                rolling_column
            )

        except:
            continue

    return df, rolling_features


# =========================================================
# CREATE TREND FEATURES
# =========================================================

def create_trend_features(
    df,
    safe_features
):

    """
    Create enterprise trend delta features.
    """

    trend_features = []

    for column in safe_features:

        try:

            trend_column = (
                f"{column}_trend_delta"
            )

            df[trend_column] = (

                df[column]

                .diff()
            )

            trend_features.append(
                trend_column
            )

        except:
            continue

    return df, trend_features


# =========================================================
# REMOVE HIGHLY CORRELATED FEATURES
# =========================================================

def remove_high_correlation_features(
    df,
    protected_targets,
    threshold=0.98
):

    """
    Remove duplicate-like features.
    """

    numeric_df = df.select_dtypes(

        include=np.number
    )

    correlation_matrix = (
        numeric_df.corr().abs()
    )

    upper_triangle = correlation_matrix.where(

        np.triu(

            np.ones(
                correlation_matrix.shape
            ),

            k=1
        ).astype(bool)
    )

    removable_columns = []

    for column in upper_triangle.columns:

        # =============================================
        # NEVER REMOVE TARGETS
        # =============================================

        if column in protected_targets:

            continue

        if any(

            upper_triangle[column] > threshold
        ):

            removable_columns.append(
                column
            )

    df.drop(

        columns=removable_columns,

        inplace=True,

        errors="ignore"
    )

    return df, removable_columns


# =========================================================
# FINAL CLEANUP
# =========================================================

def final_feature_cleanup(df):

    """
    Final engineered feature cleanup.
    """

    # =====================================================
    # REMOVE INF
    # =====================================================

    df.replace(

        [np.inf, -np.inf],

        0,

        inplace=True
    )

    # =====================================================
    # FILL NAN
    # =====================================================

    df.fillna(

        0,

        inplace=True
    )

    return df


# =========================================================
# BUILD ENTERPRISE FEATURES
# =========================================================

def build_features(df):

    """
    Enterprise feature engineering pipeline.
    """

    print(
        "\nStarting Enterprise Feature Engineering..."
    )

    # =====================================================
    # COPY
    # =====================================================

    engineered_df = df.copy()

    # =====================================================
    # DETECT TARGETS
    # =====================================================

    targets = detect_enterprise_targets(
        engineered_df
    )

    protected_targets = [

        value

        for value in targets.values()

        if value is not None
    ]

    print(
        f"Protected Targets: {protected_targets}"
    )

    # =====================================================
    # SAFE FEATURES
    # =====================================================

    safe_features = get_safe_numeric_features(

        engineered_df,

        protected_targets
    )

    print(
        f"Safe Features: {len(safe_features)}"
    )

    # =====================================================
    # LAG FEATURES
    # =====================================================

    engineered_df, lag_features = (

        create_lag_features(

            engineered_df,

            safe_features
        )
    )

    # =====================================================
    # ROLLING FEATURES
    # =====================================================

    engineered_df, rolling_features = (

        create_rolling_features(

            engineered_df,

            safe_features
        )
    )

    # =====================================================
    # TREND FEATURES
    # =====================================================

    engineered_df, trend_features = (

        create_trend_features(

            engineered_df,

            safe_features
        )
    )

    # =====================================================
    # REMOVE HIGH CORRELATION
    # =====================================================

    engineered_df, removed_columns = (

        remove_high_correlation_features(

            engineered_df,

            protected_targets
        )
    )

    # =====================================================
    # FINAL CLEANUP
    # =====================================================

    engineered_df = final_feature_cleanup(
        engineered_df
    )

    print(
        "Enterprise Feature Engineering Completed!"
    )

    # =====================================================
    # FEATURE SUMMARY
    # =====================================================

    generated_features = (

        lag_features

        +

        rolling_features

        +

        trend_features
    )

    # =====================================================
    # RETURN
    # =====================================================

    return {

        "engineered_df":
        engineered_df,

        "detected_targets":
        targets,

        "protected_targets":
        protected_targets,

        "safe_features":
        safe_features,

        "generated_features":
        generated_features,

        "removed_correlated_features":
        removed_columns
    }