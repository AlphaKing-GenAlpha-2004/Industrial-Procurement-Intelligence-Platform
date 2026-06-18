# =========================================================
# validation.py
# FINAL ENTERPRISE PROCUREMENT VALIDATION ENGINE
# FAST + CLEAN + PROCUREMENT INTELLIGENCE READY
# =========================================================

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# IMPORTS
# =========================================================

import numpy as np
import pandas as pd

# =========================================================
# DATASET SHAPE
# =========================================================

def validate_dataset_shape(df):

    return {

        "rows":
        int(df.shape[0]),

        "columns":
        int(df.shape[1]),

        "is_empty":
        bool(df.empty)
    }

# =========================================================
# MISSING VALUES
# =========================================================

def validate_missing_values(df):

    missing_counts = df.isnull().sum()

    missing_percentages = (

        missing_counts

        /

        max(len(df), 1)

    ) * 100

    missing_df = pd.DataFrame({

        "Column":
        missing_counts.index,

        "Missing Values":
        missing_counts.values,

        "Missing Percentage":
        np.round(

            missing_percentages.values,

            2
        )
    })

    missing_df = missing_df.sort_values(

        by="Missing Percentage",

        ascending=False
    )

    return {

        "total_missing":
        int(missing_counts.sum()),

        "columns_with_missing":
        int(
            (missing_counts > 0).sum()
        ),

        "missing_summary":
        missing_df
    }

# =========================================================
# DUPLICATES
# =========================================================

def validate_duplicates(df):

    duplicate_count = int(
        df.duplicated().sum()
    )

    duplicate_percentage = round(

        (
            duplicate_count

            /

            max(len(df), 1)
        ) * 100,

        2
    )

    return {

        "duplicate_rows":
        duplicate_count,

        "duplicate_percentage":
        duplicate_percentage
    }

# =========================================================
# DATA TYPES
# =========================================================

def validate_data_types(df):

    dtype_summary = {}

    for col in df.columns:

        dtype_summary[col] = str(
            df[col].dtype
        )

    return dtype_summary

# =========================================================
# NUMERIC ANALYSIS
# =========================================================

def validate_numeric_columns(df):

    numeric_columns = df.select_dtypes(

        include=[np.number]
    ).columns.tolist()

    numeric_summary = {}

    for col in numeric_columns:

        values = pd.to_numeric(

            df[col],

            errors="coerce"
        )

        if values.isnull().all():

            continue

        numeric_summary[col] = {

            "mean":
            round(values.mean(), 4),

            "median":
            round(values.median(), 4),

            "std":
            round(values.std(), 4),

            "min":
            round(values.min(), 4),

            "max":
            round(values.max(), 4),

            "unique_values":
            int(values.nunique()),

            "skewness":
            round(values.skew(), 4)
        }

    return {

        "numeric_column_count":
        len(numeric_columns),

        "numeric_columns":
        numeric_columns,

        "numeric_summary":
        numeric_summary
    }

# =========================================================
# CATEGORICAL ANALYSIS
# =========================================================

def validate_categorical_columns(df):

    categorical_columns = df.select_dtypes(

        include=["object"]
    ).columns.tolist()

    categorical_summary = {}

    for col in categorical_columns:

        top_values = (

            df[col]
            .astype(str)
            .value_counts()
            .head(5)
            .to_dict()
        )

        categorical_summary[col] = {

            "unique_categories":
            int(
                df[col].nunique()
            ),

            "top_categories":
            top_values
        }

    return {

        "categorical_column_count":
        len(categorical_columns),

        "categorical_columns":
        categorical_columns,

        "categorical_summary":
        categorical_summary
    }

# =========================================================
# OUTLIER DETECTION
# =========================================================

def validate_outliers(df):

    numeric_columns = df.select_dtypes(

        include=[np.number]
    ).columns.tolist()

    outlier_summary = {}

    for col in numeric_columns:

        values = pd.to_numeric(

            df[col],

            errors="coerce"
        )

        q1 = values.quantile(0.25)

        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)

        upper_bound = q3 + (1.5 * iqr)

        outliers = values[

            (values < lower_bound)

            |

            (values > upper_bound)
        ]

        outlier_summary[col] = {

            "outlier_count":
            int(len(outliers)),

            "outlier_percentage":
            round(

                (
                    len(outliers)

                    /

                    max(len(values), 1)
                ) * 100,

                2
            )
        }

    return outlier_summary

# =========================================================
# HIGH CARDINALITY
# =========================================================

def validate_high_cardinality(df):

    high_cardinality_columns = []

    for col in df.columns:

        if df[col].dtype == "object":

            unique_ratio = (

                df[col].nunique()

                /

                max(len(df), 1)
            )

            if unique_ratio > 0.80:

                high_cardinality_columns.append(col)

    return {

        "high_cardinality_columns":
        high_cardinality_columns,

        "count":
        len(high_cardinality_columns)
    }

# =========================================================
# CONSTANT COLUMNS
# =========================================================

def validate_constant_columns(df):

    constant_columns = []

    for col in df.columns:

        if df[col].nunique() <= 1:

            constant_columns.append(col)

    return {

        "constant_columns":
        constant_columns,

        "count":
        len(constant_columns)
    }

# =========================================================
# TARGET DETECTION
# =========================================================

def validate_target_candidates(df):

    candidate_targets = []

    keywords = [

        "demand",
        "sales",
        "inventory",
        "orders",
        "quantity",
        "stock",

        "cost",
        "price",
        "expense",
        "revenue",
        "profit",

        "risk",
        "failure",
        "defect",
        "safety"
    ]

    for col in df.columns:

        lower_col = str(col).lower()

        if any(

            keyword in lower_col

            for keyword in keywords
        ):

            candidate_targets.append(col)

    numeric_columns = df.select_dtypes(

        include=[np.number]
    ).columns.tolist()

    return {

        "candidate_targets":
        candidate_targets,

        "numeric_candidates":

        [

            col

            for col in candidate_targets

            if col in numeric_columns
        ]
    }

# =========================================================
# CORRELATION ANALYSIS
# =========================================================

def validate_correlations(df):

    numeric_df = df.select_dtypes(

        include=[np.number]
    )

    if numeric_df.shape[1] < 2:

        return {

            "high_correlations": {}
        }

    correlation_matrix = numeric_df.corr()

    high_correlations = {}

    for col in correlation_matrix.columns:

        strong_corr = correlation_matrix[col][

            abs(correlation_matrix[col]) > 0.80
        ]

        strong_corr = strong_corr.drop(

            labels=[col],

            errors="ignore"
        )

        if len(strong_corr) > 0:

            high_correlations[col] = strong_corr.to_dict()

    return {

        "high_correlations":
        high_correlations
    }

# =========================================================
# CLASS IMBALANCE
# =========================================================

def validate_class_imbalance(df):

    imbalance_report = {}

    for col in df.columns:

        unique_values = df[col].nunique()

        if unique_values <= 10:

            value_distribution = (

                df[col]
                .astype(str)
                .value_counts(normalize=True)
            )

            max_class_ratio = value_distribution.max()

            imbalance_report[col] = {

                "max_class_ratio":
                round(max_class_ratio, 4),

                "is_imbalanced":
                bool(max_class_ratio > 0.80)
            }

    return imbalance_report

# =========================================================
# PROCUREMENT READINESS
# =========================================================

def validate_procurement_readiness(df):

    procurement_keywords = [

        "inventory",
        "supplier",
        "vendor",
        "stock",
        "cost",
        "price",
        "orders",
        "quantity",
        "lead_time",
        "demand",
        "procurement"
    ]

    procurement_columns = []

    for col in df.columns:

        lower_col = str(col).lower()

        if any(

            keyword in lower_col

            for keyword in procurement_keywords
        ):

            procurement_columns.append(col)

    readiness_score = min(

        (
            len(procurement_columns) / 5
        ) * 100,

        100
    )

    return {

        "procurement_columns":
        procurement_columns,

        "procurement_column_count":
        len(procurement_columns),

        "procurement_readiness_score":
        round(readiness_score, 2)
    }

# =========================================================
# DATA QUALITY SCORE
# =========================================================

def calculate_data_quality_score(df):

    score = 100

    # =====================================================
    # MISSING VALUES
    # =====================================================

    missing_ratio = (

        df.isnull().sum().sum()

        /

        max(df.size, 1)
    )

    score -= missing_ratio * 30

    # =====================================================
    # DUPLICATES
    # =====================================================

    duplicate_ratio = (

        df.duplicated().sum()

        /

        max(len(df), 1)
    )

    score -= duplicate_ratio * 20

    # =====================================================
    # CONSTANT COLUMNS
    # =====================================================

    constant_columns = [

        col

        for col in df.columns

        if df[col].nunique() <= 1
    ]

    score -= len(constant_columns) * 2

    # =====================================================
    # HIGH CARDINALITY
    # =====================================================

    high_cardinality = validate_high_cardinality(
        df
    )

    score -= high_cardinality["count"] * 1.5

    score = max(score, 0)

    return round(score, 2)

# =========================================================
# MASTER VALIDATION PIPELINE
# =========================================================

def validate_dataset(df):

    validation_report = {

        # =================================================
        # BASIC
        # =================================================

        "dataset_shape":

        validate_dataset_shape(df),

        # =================================================
        # MISSING VALUES
        # =================================================

        "missing_values":

        validate_missing_values(df),

        # =================================================
        # DUPLICATES
        # =================================================

        "duplicates":

        validate_duplicates(df),

        # =================================================
        # DATA TYPES
        # =================================================

        "data_types":

        validate_data_types(df),

        # =================================================
        # NUMERIC
        # =================================================

        "numeric_analysis":

        validate_numeric_columns(df),

        # =================================================
        # CATEGORICAL
        # =================================================

        "categorical_analysis":

        validate_categorical_columns(df),

        # =================================================
        # OUTLIERS
        # =================================================

        "outlier_analysis":

        validate_outliers(df),

        # =================================================
        # HIGH CARDINALITY
        # =================================================

        "high_cardinality":

        validate_high_cardinality(df),

        # =================================================
        # CONSTANT COLUMNS
        # =================================================

        "constant_columns":

        validate_constant_columns(df),

        # =================================================
        # TARGET DETECTION
        # =================================================

        "target_candidates":

        validate_target_candidates(df),

        # =================================================
        # CORRELATIONS
        # =================================================

        "correlation_analysis":

        validate_correlations(df),

        # =================================================
        # CLASS IMBALANCE
        # =================================================

        "class_imbalance":

        validate_class_imbalance(df),

        # =================================================
        # PROCUREMENT READINESS
        # =================================================

        "procurement_readiness":

        validate_procurement_readiness(df),

        # =================================================
        # DATA QUALITY
        # =================================================

        "data_quality_score":

        calculate_data_quality_score(df)
    }

    return validation_report