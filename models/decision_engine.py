# =========================================================
# decision_engine.py
# FINAL ENTERPRISE PROCUREMENT DECISION ENGINE
# FULLY DYNAMIC + SAFE VERSION
# =========================================================

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# IMPORTS
# =========================================================

import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

from config.settings import (

    TARGET_KEYWORDS,

    PROCUREMENT_SCORE_MAP,

    PROCUREMENT_RECOMMENDATIONS
)

# =========================================================
# DETECT COLUMN
# =========================================================

def detect_column(

    columns,
    keywords
):

    for col in columns:

        col_lower = str(col).lower()

        for keyword in keywords:

            if keyword in col_lower:

                return col

    return None

# =========================================================
# GENERATE DYNAMIC STATUS
# =========================================================

def generate_dynamic_status(

    values,

    labels=(
        "Low",
        "Medium",
        "High"
    )
):

    values = pd.Series(values)

    # =====================================================
    # HANDLE NULLS
    # =====================================================

    values = pd.to_numeric(

        values,

        errors="coerce"
    )

    values = values.fillna(
        values.median()
    )

    # =====================================================
    # QUANTILES
    # =====================================================

    q1 = values.quantile(0.33)

    q2 = values.quantile(0.66)

    status = []

    for value in values:

        if value >= q2:

            status.append(labels[2])

        elif value >= q1:

            status.append(labels[1])

        else:

            status.append(labels[0])

    return status

# =========================================================
# NORMALIZE SAFETY VALUES
# =========================================================

def normalize_safety_values(values):

    normalized = []

    for value in values:

        try:

            value_str = str(value).lower()

            # =================================================
            # TEXT LABELS
            # =================================================

            if "high" in value_str:

                normalized.append("High")

            elif "medium" in value_str:

                normalized.append("Medium")

            elif "low" in value_str:

                normalized.append("Low")

            else:

                normalized.append(value)

        except:

            normalized.append("Medium")

    return normalized

# =========================================================
# PROCUREMENT DECISION LOGIC
# =========================================================

def procurement_decision_logic(

    demand_status,

    cost_status,

    safety_status
):

    # =====================================================
    # IDEAL CASE
    # =====================================================

    if (

        demand_status == "High"

        and

        cost_status == "Low"

        and

        safety_status == "Low"
    ):

        return "Procure Immediately"

    # =====================================================
    # STRATEGIC
    # =====================================================

    elif (

        demand_status == "High"

        and

        cost_status in [

            "Low",
            "Medium"
        ]

        and

        safety_status in [

            "Low",
            "Medium"
        ]
    ):

        return "Strategic Procurement"

    # =====================================================
    # CONTROLLED
    # =====================================================

    elif (

        demand_status == "Medium"

        and

        safety_status != "High"
    ):

        return "Controlled Procurement"

    # =====================================================
    # RISKY
    # =====================================================

    elif (

        safety_status == "High"

        and

        demand_status == "High"
    ):

        return "Risky Procurement"

    # =====================================================
    # AVOID
    # =====================================================

    elif (

        demand_status == "Low"

        and

        cost_status == "High"

        and

        safety_status == "High"
    ):

        return "Avoid Procurement"

    # =====================================================
    # DEFAULT
    # =====================================================

    return "Monitor Inventory"

# =========================================================
# BUSINESS RECOMMENDATION
# =========================================================

def generate_business_recommendation(

    procurement_decision
):

    return PROCUREMENT_RECOMMENDATIONS.get(

        procurement_decision,

        "Monitor enterprise operational trends."
    )

# =========================================================
# VALIDATE REQUIRED COLUMNS
# =========================================================

def validate_required_columns(

    df,
    required_columns
):

    missing_columns = [

        col

        for col in required_columns

        if col not in df.columns
    ]

    if len(missing_columns) > 0:

        raise ValueError(

            f"Missing required columns: {missing_columns}"
        )

# =========================================================
# BUILD PROCUREMENT INTELLIGENCE
# =========================================================

def build_procurement_intelligence(

    df,

    demand_column=None,

    cost_column=None,

    safety_column=None
):

    enterprise_df = df.copy()

    columns = enterprise_df.columns.tolist()

    # =====================================================
    # AUTO DETECT DEMAND COLUMN
    # =====================================================

    if demand_column is None:

        demand_column = detect_column(

            columns,

            TARGET_KEYWORDS[
                "demand"
            ]
        )

    # =====================================================
    # AUTO DETECT COST COLUMN
    # =====================================================

    if cost_column is None:

        cost_column = detect_column(

            columns,

            TARGET_KEYWORDS[
                "cost"
            ]
        )

    # =====================================================
    # AUTO DETECT SAFETY COLUMN
    # =====================================================

    if safety_column is None:

        safety_column = detect_column(

            columns,

            TARGET_KEYWORDS[
                "safety"
            ]
        )

    # =====================================================
    # VALIDATION
    # =====================================================

    required_columns = [

        demand_column,

        cost_column,

        safety_column
    ]

    required_columns = [

        col

        for col in required_columns

        if col is not None
    ]

    validate_required_columns(

        enterprise_df,

        required_columns
    )

    # =====================================================
    # DEMAND STATUS
    # =====================================================

    if demand_column is not None:

        enterprise_df[
            "Demand Status"
        ] = generate_dynamic_status(

            enterprise_df[
                demand_column
            ]
        )

    else:

        enterprise_df[
            "Demand Status"
        ] = "Medium"

    # =====================================================
    # COST STATUS
    # =====================================================

    if cost_column is not None:

        enterprise_df[
            "Cost Status"
        ] = generate_dynamic_status(

            enterprise_df[
                cost_column
            ]
        )

    else:

        enterprise_df[
            "Cost Status"
        ] = "Medium"

    # =====================================================
    # SAFETY STATUS
    # =====================================================

    if safety_column is not None:

        safety_values = enterprise_df[
            safety_column
        ]

        # =================================================
        # TRY NUMERIC
        # =================================================

        try:

            numeric_values = pd.to_numeric(

                safety_values,

                errors="coerce"
            )

            if numeric_values.notna().sum() > 0:

                enterprise_df[
                    "Safety Status"
                ] = generate_dynamic_status(
                    numeric_values
                )

            else:

                enterprise_df[
                    "Safety Status"
                ] = normalize_safety_values(
                    safety_values
                )

        except:

            enterprise_df[
                "Safety Status"
            ] = normalize_safety_values(
                safety_values
            )

    else:

        enterprise_df[
            "Safety Status"
        ] = "Medium"

    # =====================================================
    # PROCUREMENT DECISION
    # =====================================================

    enterprise_df[
        "Procurement Decision"
    ] = enterprise_df.apply(

        lambda row:

        procurement_decision_logic(

            row["Demand Status"],

            row["Cost Status"],

            row["Safety Status"]
        ),

        axis=1
    )

    # =====================================================
    # BUSINESS RECOMMENDATION
    # =====================================================

    enterprise_df[
        "Business Recommendation"
    ] = enterprise_df[
        "Procurement Decision"
    ].apply(
        generate_business_recommendation
    )

    # =====================================================
    # PROCUREMENT SCORE
    # =====================================================

    enterprise_df[
        "Procurement Score"
    ] = enterprise_df[
        "Procurement Decision"
    ].map(
        PROCUREMENT_SCORE_MAP
    )

    # =====================================================
    # ENTERPRISE PRIORITY
    # =====================================================

    enterprise_df[
        "Enterprise Priority"
    ] = np.where(

        enterprise_df[
            "Procurement Score"
        ] >= 80,

        "Critical",

        np.where(

            enterprise_df[
                "Procurement Score"
            ] >= 50,

            "Important",

            "Low"
        )
    )

    # =====================================================
    # PROCUREMENT RISK LEVEL
    # =====================================================

    enterprise_df[
        "Procurement Risk Level"
    ] = np.where(

        enterprise_df[
            "Safety Status"
        ] == "High",

        "High Risk",

        np.where(

            enterprise_df[
                "Safety Status"
            ] == "Medium",

            "Moderate Risk",

            "Low Risk"
        )
    )

    # =====================================================
    # AI PROCUREMENT CONFIDENCE
    # =====================================================

    confidence_scores = []

    for _, row in enterprise_df.iterrows():

        score = 90

        if row["Safety Status"] == "High":

            score -= 25

        if row["Cost Status"] == "High":

            score -= 15

        if row["Demand Status"] == "Low":

            score -= 10

        confidence_scores.append(
            max(score, 30)
        )

    enterprise_df[
        "AI Confidence Score"
    ] = confidence_scores

    # =====================================================
    # FINAL CLEAN
    # =====================================================

    enterprise_df = enterprise_df.replace(

        [np.inf, -np.inf],

        0
    )

    enterprise_df = enterprise_df.fillna(
        "Unknown"
    )

    return enterprise_df

# =========================================================
# KPI SUMMARY
# =========================================================

def procurement_kpi_summary(df):

    summary = {

        "total_records":
        len(df),

        "procure_immediately": 0,

        "strategic_procurement": 0,

        "controlled_procurement": 0,

        "risky_procurement": 0,

        "avoid_procurement": 0
    }

    if "Procurement Decision" not in df.columns:

        return summary

    decision_counts = df[
        "Procurement Decision"
    ].value_counts()

    summary[
        "procure_immediately"
    ] = int(

        decision_counts.get(

            "Procure Immediately",

            0
        )
    )

    summary[
        "strategic_procurement"
    ] = int(

        decision_counts.get(

            "Strategic Procurement",

            0
        )
    )

    summary[
        "controlled_procurement"
    ] = int(

        decision_counts.get(

            "Controlled Procurement",

            0
        )
    )

    summary[
        "risky_procurement"
    ] = int(

        decision_counts.get(

            "Risky Procurement",

            0
        )
    )

    summary[
        "avoid_procurement"
    ] = int(

        decision_counts.get(

            "Avoid Procurement",

            0
        )
    )

    return summary