import pandas as pd
import numpy as np

# ==========================================
# INDUSTRIAL FEATURE ENGINEERING
# ==========================================

def build_features(df):

    print("\nBUILDING INDUSTRIAL AI FEATURES...\n")

    # ======================================
    # COPY DATAFRAME
    # ======================================

    df = df.copy()

    # ======================================
    # GET NUMERIC COLUMNS
    # ======================================

    numeric_columns = df.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    # ======================================
    # BASIC INTERACTION FEATURES
    # ======================================

    for i in range(len(numeric_columns)):

        for j in range(i + 1, len(numeric_columns)):

            col1 = numeric_columns[i]
            col2 = numeric_columns[j]

            feature_name = (
                f"{col1}_{col2}_interaction"
            )

            df[feature_name] = (
                df[col1] * df[col2]
            )

    # ======================================
    # INDUSTRIAL DOMAIN FEATURES
    # ======================================

    # --------------------------------------
    # INVENTORY TO DEMAND RATIO
    # --------------------------------------

    if (
        "inventory_level" in df.columns
        and
        "historical_demand" in df.columns
    ):

        df["inventory_to_demand_ratio"] = (

            df["inventory_level"]
            /
            (
                df["historical_demand"] + 1
            )
        )

    # --------------------------------------
    # COST PER LABOR HOUR
    # --------------------------------------

    if (
        "production_cost" in df.columns
        and
        "labor_hours" in df.columns
    ):

        df["cost_per_labor_hour"] = (

            df["production_cost"]
            /
            (
                df["labor_hours"] + 1
            )
        )

    # --------------------------------------
    # SUPPLIER EFFICIENCY
    # --------------------------------------

    if (
        "supplier_rating" in df.columns
        and
        "lead_time" in df.columns
    ):

        df["supplier_efficiency"] = (

            df["supplier_rating"]
            /
            (
                df["lead_time"] + 1
            )
        )

    # --------------------------------------
    # MACHINE LOAD INDEX
    # --------------------------------------

    if (
        "machine_utilization" in df.columns
        and
        "labor_hours" in df.columns
    ):

        df["machine_load_index"] = (

            df["machine_utilization"]
            *
            df["labor_hours"]
        )

    # --------------------------------------
    # PRESSURE TEMPERATURE INDEX
    # --------------------------------------

    if (
        "pressure" in df.columns
        and
        "temperature" in df.columns
    ):

        df["pressure_temperature_index"] = (

            df["pressure"]
            *
            df["temperature"]
        )

    # --------------------------------------
    # COST DEMAND RATIO
    # --------------------------------------

    if (
        "production_cost" in df.columns
        and
        "historical_demand" in df.columns
    ):

        df["cost_demand_ratio"] = (

            df["production_cost"]
            /
            (
                df["historical_demand"] + 1
            )
        )

    # --------------------------------------
    # MATERIAL STRESS INDEX
    # --------------------------------------

    if (
        "pressure" in df.columns
        and
        "machine_utilization" in df.columns
    ):

        df["material_stress_index"] = (

            df["pressure"]
            *
            df["machine_utilization"]
        )

    # --------------------------------------
    # SAFETY LOAD INDEX
    # --------------------------------------

    if (
        "temperature" in df.columns
        and
        "pressure" in df.columns
        and
        "machine_utilization" in df.columns
    ):

        df["safety_load_index"] = (

            (
                df["temperature"]
                *
                df["pressure"]
            )
            *
            df["machine_utilization"]
        )

    # ======================================
    # POLYNOMIAL FEATURES
    # ======================================

    important_columns = [

        "historical_demand",
        "production_cost",
        "inventory_level",
        "labor_hours"
    ]

    for col in important_columns:

        if col in df.columns:

            df[f"{col}_squared"] = (

                df[col] ** 2
            )

            df[f"{col}_log"] = np.log1p(
                df[col]
            )

    # ======================================
    # ROLLING FEATURES
    # ======================================

    rolling_columns = [

        "historical_demand",
        "production_cost"
    ]

    for col in rolling_columns:

        if col in df.columns:

            df[f"{col}_rolling_mean"] = (

                df[col]
                .rolling(
                    window=5,
                    min_periods=1
                )
                .mean()
            )

            df[f"{col}_rolling_std"] = (

                df[col]
                .rolling(
                    window=5,
                    min_periods=1
                )
                .std()
                .fillna(0)
            )

    # ======================================
    # LAG FEATURES
    # ======================================

    lag_columns = [

        "historical_demand",
        "production_cost"
    ]

    for col in lag_columns:

        if col in df.columns:

            df[f"{col}_lag_1"] = (
                df[col]
                .shift(1)
                .fillna(df[col].mean())
            )

            df[f"{col}_lag_2"] = (
                df[col]
                .shift(2)
                .fillna(df[col].mean())
            )

    # ======================================
    # REMOVE INFINITE VALUES
    # ======================================

    df.replace(
        [np.inf, -np.inf],
        0,
        inplace=True
    )

    # ======================================
    # FILL REMAINING NaNs
    # ======================================

    df.fillna(
        0,
        inplace=True
    )

    print(
        "Feature Engineering Completed!"
    )

    print(
        f"Total Features: {df.shape[1]}"
    )

    return df