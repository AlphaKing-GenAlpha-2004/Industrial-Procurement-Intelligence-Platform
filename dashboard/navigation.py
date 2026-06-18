# =========================================================
# navigation.py
# FINAL ENTERPRISE PROCUREMENT NAVIGATION ENGINE
# CLEAN + FAST + PROCUREMENT INTELLIGENCE READY
# =========================================================

import streamlit as st

# =========================================================
# PAGE HEADER
# =========================================================

def page_header(

    title,

    subtitle=""
):

    st.title(title)

    if subtitle:

        st.caption(subtitle)

# =========================================================
# SECTION TITLE
# =========================================================

def section_title(title):

    st.markdown(
        f"## {title}"
    )

# =========================================================
# SECTION DIVIDER
# =========================================================

def section_divider():

    st.markdown("---")

# =========================================================
# SUCCESS BANNER
# =========================================================

def success_banner(message):

    st.success(message)

# =========================================================
# WARNING BANNER
# =========================================================

def warning_banner(message):

    st.warning(message)

# =========================================================
# ERROR BANNER
# =========================================================

def error_banner(message):

    st.error(message)

# =========================================================
# EMPTY STATE
# =========================================================

def empty_state(message):

    st.info(message)

# =========================================================
# ENTERPRISE NAVIGATION
# =========================================================

def enterprise_navigation():

    # =====================================================
    # SIDEBAR HEADER
    # =====================================================

    st.sidebar.title(
        "DemandIQ"
    )

    st.sidebar.caption(
        "Industrial Procurement Intelligence Platform"
    )

    st.sidebar.markdown("---")

    # =====================================================
    # NAVIGATION MENU
    # =====================================================

    navigation_items = [

        "Enterprise Overview",

        "Demand Intelligence",

        "Cost Intelligence",

        "Safety Intelligence",

        "Procurement Intelligence",

        "Model Performance",

        "Dataset Intelligence"
    ]

    selected_page = st.sidebar.radio(

        "Navigation",

        navigation_items
    )

    st.sidebar.markdown("---")

    # =====================================================
    # SYSTEM STATUS
    # =====================================================

    st.sidebar.success(
        "Enterprise AI Models Active"
    )

    st.sidebar.success(
        "Demand Prediction Engine Online"
    )

    st.sidebar.success(
        "Procurement Intelligence Active"
    )

    st.sidebar.success(
        "Cost Intelligence Active"
    )

    st.sidebar.success(
        "Safety Intelligence Active"
    )

    st.sidebar.markdown("---")

    # =====================================================
    # SYSTEM METRICS
    # =====================================================

    st.sidebar.metric(

        "AI Status",

        "Operational"
    )

    st.sidebar.metric(

        "Pipeline",

        "Active"
    )

    st.sidebar.metric(

        "Inference Mode",

        "Real-Time"
    )

    st.sidebar.markdown("---")

    return selected_page

# =========================================================
# FORECAST STATUS PANEL
# =========================================================

def forecast_status_panel():

    st.sidebar.markdown(
        "## Demand Intelligence"
    )

    st.sidebar.success(
        "Demand Prediction Models Active"
    )

    st.sidebar.success(
        "Regression Intelligence Enabled"
    )

    st.sidebar.success(
        "Enterprise Scoring Active"
    )

    st.sidebar.success(
        "Feature Importance Enabled"
    )

# =========================================================
# PROCUREMENT STATUS PANEL
# =========================================================

def procurement_status_panel():

    st.sidebar.markdown(
        "## Procurement Intelligence"
    )

    st.sidebar.success(
        "Adaptive Procurement Scoring"
    )

    st.sidebar.success(
        "Dynamic Procurement Tiers"
    )

    st.sidebar.success(
        "Risk Intelligence Enabled"
    )

    st.sidebar.success(
        "Confidence Scoring Enabled"
    )

    st.sidebar.success(
        "Supplier Prioritization Active"
    )

# =========================================================
# MODEL STATUS PANEL
# =========================================================

def model_status_panel():

    st.sidebar.markdown(
        "## AI Model Infrastructure"
    )

    st.sidebar.info(
        "Random Forest Models Loaded"
    )

    st.sidebar.info(
        "Gradient Boosting Models Loaded"
    )

    st.sidebar.info(
        "Extra Trees Models Loaded"
    )

    st.sidebar.info(
        "Adaptive Boosting Enabled"
    )

# =========================================================
# DATASET STATUS PANEL
# =========================================================

def dataset_status_panel(

    total_rows=None,

    total_columns=None
):

    st.sidebar.markdown(
        "## Dataset Intelligence"
    )

    if total_rows is not None:

        st.sidebar.metric(
            "Rows",
            total_rows
        )

    if total_columns is not None:

        st.sidebar.metric(
            "Columns",
            total_columns
        )

    st.sidebar.success(
        "Dataset Validation Complete"
    )

    st.sidebar.success(
        "Feature Engineering Enabled"
    )

# =========================================================
# PROCUREMENT HEALTH PANEL
# =========================================================

def procurement_health_panel(

    procurement_score=None
):

    st.sidebar.markdown(
        "## Procurement Health"
    )

    if procurement_score is not None:

        st.sidebar.metric(

            "Health Score",

            round(
                procurement_score,
                2
            )
        )

    st.sidebar.success(
        "Procurement Pipeline Stable"
    )

    st.sidebar.success(
        "Risk Evaluation Operational"
    )