# =========================================================
# kpi_cards.py
# MINIMAL SAFE VERSION
# GUARANTEED WORKING
# NO HTML BREAKS
# =========================================================

import streamlit as st

# =========================================================
# MAIN KPI CARD
# =========================================================

def kpi_card(

    title,
    value,
    subtitle="",
    icon="📊"
):

    st.metric(

        label=f"{icon} {title}",

        value=value,

        delta=subtitle
    )

# =========================================================
# KPI ROW
# =========================================================

def enterprise_kpi_row(kpi_data):

    if len(kpi_data) == 0:

        return

    cols = st.columns(len(kpi_data))

    for idx, kpi in enumerate(kpi_data):

        with cols[idx]:

            kpi_card(

                title=kpi.get(
                    "title",
                    ""
                ),

                value=kpi.get(
                    "value",
                    ""
                ),

                subtitle=kpi.get(
                    "subtitle",
                    ""
                ),

                icon=kpi.get(
                    "icon",
                    "📊"
                )
            )

# =========================================================
# PROCUREMENT KPI SECTION
# =========================================================

def procurement_kpi_section(summary):

    kpi_data = [

        {

            "title":
            "Total Records",

            "value":
            f"{summary.get('total_records', 0):,}",

            "subtitle":
            "Operational Records",

            "icon":
            "📦"
        },

        {

            "title":
            "Immediate Procurement",

            "value":
            summary.get(
                "procure_immediately",
                0
            ),

            "subtitle":
            "High Priority",

            "icon":
            "🚀"
        },

        {

            "title":
            "Strategic Procurement",

            "value":
            summary.get(
                "strategic_procurement",
                0
            ),

            "subtitle":
            "Optimization",

            "icon":
            "📈"
        },

        {

            "title":
            "Risky Procurement",

            "value":
            summary.get(
                "risky_procurement",
                0
            ),

            "subtitle":
            "Risk Detected",

            "icon":
            "⚠️"
        },

        {

            "title":
            "Avoid Procurement",

            "value":
            summary.get(
                "avoid_procurement",
                0
            ),

            "subtitle":
            "Avoid Items",

            "icon":
            "❌"
        }
    ]

    enterprise_kpi_row(kpi_data)

# =========================================================
# EXECUTIVE SUMMARY
# =========================================================

def executive_summary_row(

    total_records,
    avg_confidence,
    avg_score,
    high_risk_items
):

    kpi_data = [

        {

            "title":
            "Enterprise Records",

            "value":
            f"{total_records:,}",

            "subtitle":
            "Processed",

            "icon":
            "🏢"
        },

        {

            "title":
            "AI Confidence",

            "value":
            f"{avg_confidence:.1f}%",

            "subtitle":
            "Prediction Confidence",

            "icon":
            "🧠"
        },

        {

            "title":
            "Procurement Score",

            "value":
            f"{avg_score:.1f}",

            "subtitle":
            "Optimization Score",

            "icon":
            "📈"
        },

        {

            "title":
            "High Risk Items",

            "value":
            high_risk_items,

            "subtitle":
            "Critical Risks",

            "icon":
            "⚠️"
        }
    ]

    enterprise_kpi_row(kpi_data)

# =========================================================
# SELECTED MODEL CARD
# =========================================================

def selected_model_card(

    target_name,
    model_name,
    score
):

    st.info(

        f"""
        {target_name}

        Best Model:
        {model_name}

        Score:
        {round(score, 4)}
        """
    )

# =========================================================
# DATASET OVERVIEW CARD
# =========================================================

def dataset_overview_card(

    rows,
    columns,
    missing_values
):

    st.success(

        f"""
        Dataset Rows:
        {rows:,}

        Columns:
        {columns}

        Missing Values:
        {missing_values}
        """
    )

# =========================================================
# MINI KPI CARD
# =========================================================

def mini_kpi_card(

    title,
    value
):

    st.metric(

        label=title,

        value=value
    )

# =========================================================
# STATUS CARD
# =========================================================

def status_card(

    title,
    status
):

    st.info(

        f"{title}: {status}"
    )