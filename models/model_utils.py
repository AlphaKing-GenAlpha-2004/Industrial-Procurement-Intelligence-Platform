# =========================================================
# model_utils.py
# ENTERPRISE MODEL UTILITIES
# STRICT TARGET DETECTION + SAFE UTILITIES
# =========================================================

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# IMPORTS
# =========================================================

import numpy as np
import pandas as pd

# =========================================================
# ENTERPRISE TARGET DETECTION
# =========================================================

def detect_enterprise_targets(df):

    """
    Strict enterprise target detection.

    Prevents:
    - wrong categorical targets
    - random material columns
    - supplier names
    - status columns

    Ensures:
    - proper demand targets
    - proper cost targets
    - proper safety targets
    """

    targets = {

        "demand_target": None,

        "cost_target": None,

        "safety_target": None
    }

    # =====================================================
    # CLEAN COLUMN LIST
    # =====================================================

    columns = [

        str(col).lower()

        for col in df.columns
    ]

    # =====================================================
    # STRICT PRIORITY TARGETS
    # =====================================================

    demand_priority = [

        "historical_demand",

        "forecast_demand",

        "predicted_demand",

        "market_demand",

        "customer_demand",

        "demand_forecast",

        "sales_volume",

        "sales",

        "demand"
    ]

    cost_priority = [

        "procurement_cost",

        "raw_material_cost",

        "transportation_cost",

        "holding_cost",

        "maintenance_cost",

        "operational_cost",

        "logistics_cost",

        "fuel_cost",

        "inventory_cost",

        "production_cost",

        "cost"
    ]

    safety_priority = [

        "safety_risk",

        "risk_level",

        "risk_status",

        "risk_category",

        "safety_level",

        "risk",

        "safety"
    ]

    # =====================================================
    # DEMAND TARGET
    # =====================================================

    for keyword in demand_priority:

        for column in df.columns:

            lower_column = str(column).lower()

            # =============================================
            # MATCH
            # =============================================

            if keyword == lower_column:

                # =========================================
                # MUST BE NUMERIC
                # =========================================

                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):

                    targets[
                        "demand_target"
                    ] = column

                    break

        if targets["demand_target"] is not None:

            break

    # =====================================================
    # SECONDARY DEMAND SEARCH
    # =====================================================

    if targets["demand_target"] is None:

        for keyword in demand_priority:

            for column in df.columns:

                lower_column = str(column).lower()

                if keyword in lower_column:

                    if pd.api.types.is_numeric_dtype(
                        df[column]
                    ):

                        targets[
                            "demand_target"
                        ] = column

                        break

            if targets["demand_target"] is not None:

                break

    # =====================================================
    # COST TARGET
    # =====================================================

    for keyword in cost_priority:

        for column in df.columns:

            lower_column = str(column).lower()

            # =============================================
            # STRICT MATCH
            # =============================================

            if keyword == lower_column:

                # =========================================
                # MUST BE NUMERIC
                # =========================================

                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):

                    targets[
                        "cost_target"
                    ] = column

                    break

        if targets["cost_target"] is not None:

            break

    # =====================================================
    # SECONDARY COST SEARCH
    # =====================================================

    if targets["cost_target"] is None:

        for keyword in cost_priority:

            for column in df.columns:

                lower_column = str(column).lower()

                if keyword in lower_column:

                    # =====================================
                    # MUST BE NUMERIC
                    # =====================================

                    if pd.api.types.is_numeric_dtype(
                        df[column]
                    ):

                        # =================================
                        # AVOID BAD TARGETS
                        # =================================

                        blocked_keywords = [

                            "material_type",

                            "material_category",

                            "material_name",

                            "supplier",

                            "status",

                            "region",

                            "category"
                        ]

                        if any(

                            blocked in lower_column

                            for blocked in blocked_keywords
                        ):

                            continue

                        targets[
                            "cost_target"
                        ] = column

                        break

            if targets["cost_target"] is not None:

                break

    # =====================================================
    # SAFETY TARGET
    # =====================================================

    for keyword in safety_priority:

        for column in df.columns:

            lower_column = str(column).lower()

            # =============================================
            # STRICT MATCH
            # =============================================

            if keyword == lower_column:

                targets[
                    "safety_target"
                ] = column

                break

        if targets["safety_target"] is not None:

            break

    # =====================================================
    # SECONDARY SAFETY SEARCH
    # =====================================================

    if targets["safety_target"] is None:

        for keyword in safety_priority:

            for column in df.columns:

                lower_column = str(column).lower()

                if keyword in lower_column:

                    # =====================================
                    # AVOID BAD TARGETS
                    # =====================================

                    blocked_keywords = [

                        "machine_status",

                        "procurement_status",

                        "material_status",

                        "transport_status"
                    ]

                    if any(

                        blocked in lower_column

                        for blocked in blocked_keywords
                    ):

                        continue

                    targets[
                        "safety_target"
                    ] = column

                    break

            if targets["safety_target"] is not None:

                break

    # =====================================================
    # VALIDATION
    # =====================================================

    # DEMAND MUST BE NUMERIC
    if targets["demand_target"] is not None:

        if not pd.api.types.is_numeric_dtype(

            df[
                targets["demand_target"]
            ]
        ):

            targets["demand_target"] = None

    # COST MUST BE NUMERIC
    if targets["cost_target"] is not None:

        if not pd.api.types.is_numeric_dtype(

            df[
                targets["cost_target"]
            ]
        ):

            targets["cost_target"] = None

    # SAFETY SHOULD BE CLASSIFICATION
    if targets["safety_target"] is not None:

        unique_values = (

            df[
                targets["safety_target"]
            ]

            .astype(str)

            .nunique()
        )

        # =============================================
        # TOO MANY CLASSES = BAD TARGET
        # =============================================

        if unique_values > 20:

            targets["safety_target"] = None

    return targets


# =========================================================
# SAFE FEATURE IMPORTANCE
# =========================================================

def safe_feature_importance(

    model,

    feature_names
):

    """
    Generate safe feature importance dataframe.
    """

    try:

        # =================================================
        # FEATURE IMPORTANCE
        # =================================================

        if hasattr(

            model,

            "feature_importances_"
        ):

            importance_values = (
                model.feature_importances_
            )

        else:

            return None

        # =================================================
        # DATAFRAME
        # =================================================

        importance_df = pd.DataFrame({

            "Feature":
            feature_names,

            "Importance":
            importance_values
        })

        # =================================================
        # SORT
        # =================================================

        importance_df = (

            importance_df

            .sort_values(

                by="Importance",

                ascending=False
            )

            .reset_index(drop=True)
        )

        # =================================================
        # TOP FEATURES ONLY
        # =================================================

        importance_df = (
            importance_df.head(15)
        )

        return importance_df

    except:

        return None


# =========================================================
# SAFE NUMERIC COLUMN DETECTION
# =========================================================

def safe_numeric_columns(df):

    """
    Safely detect numeric columns.
    """

    try:

        numeric_columns = (

            df.select_dtypes(

                include=np.number
            )

            .columns

            .tolist()
        )

        return numeric_columns

    except:

        return []


# =========================================================
# SAFE CATEGORICAL COLUMN DETECTION
# =========================================================

def safe_categorical_columns(df):

    """
    Safely detect categorical columns.
    """

    try:

        categorical_columns = (

            df.select_dtypes(

                include=["object"]
            )

            .columns

            .tolist()
        )

        return categorical_columns

    except:

        return []


# =========================================================
# REMOVE BAD FEATURES
# =========================================================

def remove_bad_features(df):

    """
    Remove unstable enterprise features.
    """

    removable_keywords = [

        "_interaction",

        "_ratio",

        "_polynomial",

        "_complex",

        "_sin",

        "_cos"
    ]

    removable_columns = []

    for column in df.columns:

        lower_column = str(column).lower()

        if any(

            keyword in lower_column

            for keyword in removable_keywords
        ):

            removable_columns.append(
                column
            )

    cleaned_df = df.drop(

        columns=removable_columns,

        errors="ignore"
    )

    return cleaned_df