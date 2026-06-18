# =========================================================
# theme.py
# FINAL ENTERPRISE UI THEME ENGINE
# CLEAN + PROFESSIONAL + SAFE
# =========================================================

import streamlit as st

# =========================================================
# LOAD ENTERPRISE THEME
# =========================================================

def load_enterprise_theme():

    st.markdown(

        """
        <style>

        /* =================================================
        GLOBAL APP
        ================================================= */

        .stApp {

            background:
            linear-gradient(
                135deg,
                #020617 0%,
                #081225 50%,
                #0f172a 100%
            );

            color: #f8fafc;
        }

        /* =================================================
        MAIN CONTAINER
        ================================================= */

        .main .block-container {

            max-width: 100%;

            padding-top: 2rem;

            padding-left: 2rem;

            padding-right: 2rem;

            padding-bottom: 2rem;
        }

        /* =================================================
        SIDEBAR
        ================================================= */

        section[data-testid="stSidebar"] {

            background:
            linear-gradient(
                180deg,
                #081225 0%,
                #0f172a 100%
            );

            border-right:
            1px solid rgba(255,255,255,0.06);
        }

        /* =================================================
        HEADINGS
        ================================================= */

        h1, h2, h3, h4, h5 {

            color: #ffffff !important;

            font-weight: 700;
        }

        /* =================================================
        TEXT
        ================================================= */

        p, span, label, div {

            color: #e2e8f0;
        }

        /* =================================================
        METRIC CARDS
        ================================================= */

        div[data-testid="metric-container"] {

            background:
            rgba(255,255,255,0.04);

            border:
            1px solid rgba(255,255,255,0.08);

            border-radius: 18px;

            padding: 18px;

            box-shadow:
            0 8px 28px rgba(0,0,0,0.30);

            transition: 0.3s ease-in-out;
        }

        div[data-testid="metric-container"]:hover {

            transform: translateY(-2px);

            border:
            1px solid rgba(0,245,160,0.25);
        }

        /* =================================================
        BUTTONS
        ================================================= */

        .stButton > button {

            background:
            linear-gradient(
                90deg,
                #00f5a0,
                #00d9f5
            );

            color: #000000;

            border: none;

            border-radius: 12px;

            font-weight: 700;

            padding:
            0.6rem 1.5rem;

            transition: 0.3s ease;
        }

        .stButton > button:hover {

            transform: scale(1.02);

            opacity: 0.95;
        }

        /* =================================================
        DOWNLOAD BUTTON
        ================================================= */

        .stDownloadButton > button {

            background:
            linear-gradient(
                90deg,
                #38bdf8,
                #0ea5e9
            );

            color: white;

            border: none;

            border-radius: 12px;

            font-weight: 700;

            padding:
            0.6rem 1.5rem;
        }

        /* =================================================
        DATAFRAME
        ================================================= */

        .stDataFrame {

            border-radius: 18px;

            overflow: hidden;

            border:
            1px solid rgba(255,255,255,0.08);

            background:
            rgba(255,255,255,0.02);
        }

        /* =================================================
        FILE UPLOADER
        ================================================= */

        section[data-testid="stFileUploader"] {

            background:
            rgba(255,255,255,0.03);

            padding: 18px;

            border-radius: 18px;

            border:
            1px solid rgba(255,255,255,0.08);
        }

        /* =================================================
        PLOTLY CHARTS
        ================================================= */

        .js-plotly-plot {

            border-radius: 18px;

            overflow: hidden;
        }

        /* =================================================
        ALERT BOXES
        ================================================= */

        div[data-baseweb="notification"] {

            border-radius: 16px;
        }

        /* =================================================
        SCROLLBAR
        ================================================= */

        ::-webkit-scrollbar {

            width: 10px;
        }

        ::-webkit-scrollbar-thumb {

            background:
            rgba(255,255,255,0.18);

            border-radius: 10px;
        }

        ::-webkit-scrollbar-track {

            background:
            transparent;
        }

        </style>
        """,

        unsafe_allow_html=True
    )

# =========================================================
# HERO HEADER
# =========================================================

def hero_header():

    st.markdown(

        """
        # Industrial AI Decision Intelligence System
        """,

        unsafe_allow_html=True
    )

    st.caption(
        "Enterprise Procurement Analytics • Demand Prediction • Cost Prediction • Safety Intelligence • Operational Intelligence"
    )

    st.markdown("---")

# =========================================================
# SECTION TITLE
# =========================================================

def section_title(title):

    st.markdown(
        f"## {title}"
    )

# =========================================================
# STATUS CHIP
# =========================================================

def status_chip(

    text,

    color="green"
):

    if color.lower() == "green":

        st.success(text)

    elif color.lower() == "red":

        st.error(text)

    elif color.lower() == "orange":

        st.warning(text)

    elif color.lower() == "blue":

        st.info(text)

    else:

        st.info(text)

# =========================================================
# KPI CARD
# =========================================================

def kpi_card(

    title,

    value,

    delta=None
):

    st.metric(

        label=title,

        value=value,

        delta=delta
    )

# =========================================================
# INFO PANEL
# =========================================================

def info_panel(

    title,

    content
):

    st.markdown(

        f"""
        <div style="
            padding:18px;
            border-radius:18px;
            background:rgba(255,255,255,0.03);
            border:1px solid rgba(255,255,255,0.08);
            margin-bottom:16px;
        ">
            <h4>{title}</h4>
            <p>{content}</p>
        </div>
        """,

        unsafe_allow_html=True
    )