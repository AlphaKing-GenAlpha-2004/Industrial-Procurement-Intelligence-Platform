# =========================================================
# plotly_charts.py
# FINAL ENTERPRISE PROCUREMENT VISUALIZATION ENGINE
# CLEAN + FAST + STABLE + PROFESSIONAL
# =========================================================

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# IMPORTS
# =========================================================

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# ENTERPRISE THEME
# =========================================================

ENTERPRISE_TEMPLATE = "plotly_dark"

ENTERPRISE_HEIGHT = 500

# =========================================================
# ENTERPRISE STYLING
# =========================================================

def apply_enterprise_style(fig):

    fig.update_layout(

        template=ENTERPRISE_TEMPLATE,

        height=ENTERPRISE_HEIGHT,

        title_x=0.5,

        font=dict(

            family="Arial",

            size=14,

            color="#f8fafc"
        ),

        paper_bgcolor="#020617",

        plot_bgcolor="#020617",

        margin=dict(

            l=40,

            r=40,

            t=80,

            b=40
        ),

        legend=dict(

            orientation="h",

            yanchor="bottom",

            y=1.02,

            xanchor="right",

            x=1
        )
    )

    return fig

# =========================================================
# DEMAND PREDICTION CHART
# =========================================================

def demand_prediction_chart(

    actual_values,

    predicted_values
):

    # =====================================================
    # VALIDATION
    # =====================================================

    if (

        actual_values is None

        or

        predicted_values is None
    ):

        return None

    actual_values = np.array(
        actual_values
    )

    predicted_values = np.array(
        predicted_values
    )

    if (

        len(actual_values) == 0

        or

        len(predicted_values) == 0
    ):

        return None

    # =====================================================
    # MATCH LENGTHS
    # =====================================================

    min_length = min(

        len(actual_values),

        len(predicted_values)
    )

    actual_values = actual_values[
        :min_length
    ]

    predicted_values = predicted_values[
        :min_length
    ]

    # =====================================================
    # CHART
    # =====================================================

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=list(
                range(min_length)
            ),

            y=actual_values,

            mode="lines",

            name="Actual Demand",

            line=dict(

                width=3
            )
        )
    )

    fig.add_trace(

        go.Scatter(

            x=list(
                range(min_length)
            ),

            y=predicted_values,

            mode="lines",

            name="Predicted Demand",

            line=dict(

                width=3,

                dash="dash"
            )
        )
    )

    fig.update_layout(

        title="Demand Forecast vs Actual Demand",

        xaxis_title="Record Index",

        yaxis_title="Demand Value",

        hovermode="x unified"
    )

    return apply_enterprise_style(
        fig
    )

# =========================================================
# COST PREDICTION CHART
# =========================================================

def cost_prediction_chart(

    actual_values,

    predicted_values
):

    # =====================================================
    # VALIDATION
    # =====================================================

    if (

        actual_values is None

        or

        predicted_values is None
    ):

        return None

    actual_values = np.array(
        actual_values
    )

    predicted_values = np.array(
        predicted_values
    )

    if (

        len(actual_values) == 0

        or

        len(predicted_values) == 0
    ):

        return None

    # =====================================================
    # MATCH LENGTHS
    # =====================================================

    min_length = min(

        len(actual_values),

        len(predicted_values)
    )

    actual_values = actual_values[
        :min_length
    ]

    predicted_values = predicted_values[
        :min_length
    ]

    # =====================================================
    # CHART
    # =====================================================

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=list(
                range(min_length)
            ),

            y=actual_values,

            mode="lines",

            name="Actual Cost",

            line=dict(

                width=3
            )
        )
    )

    fig.add_trace(

        go.Scatter(

            x=list(
                range(min_length)
            ),

            y=predicted_values,

            mode="lines",

            name="Predicted Cost",

            line=dict(

                width=3,

                dash="dash"
            )
        )
    )

    fig.update_layout(

        title="Cost Forecast vs Actual Cost",

        xaxis_title="Record Index",

        yaxis_title="Cost Value",

        hovermode="x unified"
    )

    return apply_enterprise_style(
        fig
    )

# =========================================================
# FEATURE IMPORTANCE CHART
# =========================================================

def feature_importance_chart(

    feature_importance_df
):

    if (

        feature_importance_df is None

        or len(feature_importance_df) == 0
    ):

        return None

    feature_importance_df = (

        feature_importance_df

        .sort_values(

            by="Importance",

            ascending=True
        )

        .tail(15)
    )

    fig = px.bar(

        feature_importance_df,

        x="Importance",

        y="Feature",

        orientation="h",

        title="Feature Importance (%)"
    )

    fig.update_layout(

        xaxis_title="Importance (%)",

        yaxis_title="Features"
    )

    return apply_enterprise_style(fig)

# =========================================================
# PROCUREMENT SCORE DISTRIBUTION
# =========================================================

def procurement_score_chart(

    procurement_scores
):

    fig = px.histogram(

        x=procurement_scores,

        nbins=25,

        title="Procurement Score Distribution"
    )

    fig.update_layout(

        xaxis_title="Procurement Score",

        yaxis_title="Frequency"
    )

    return apply_enterprise_style(fig)

# =========================================================
# SUPPLIER RECOMMENDATION CHART
# =========================================================

def supplier_ranking_chart(

    supplier_intelligence_df

):

    if (

        supplier_intelligence_df is None

        or

        supplier_intelligence_df.empty

    ):

        return None

    required_columns = [

        "Recommended Supplier",

        "Procurement Score"
    ]

    if not all(

        col in supplier_intelligence_df.columns

        for col in required_columns

    ):

        return None

    supplier_summary = (

        supplier_intelligence_df

        .groupby(
            "Recommended Supplier"
        )

        .agg({

            "Procurement Score":
            "mean"
        })

        .reset_index()

        .sort_values(

            by="Procurement Score",

            ascending=False
        )
    )

    supplier_summary[
        "Procurement Score"
    ] = supplier_summary[
        "Procurement Score"
    ].round(2)

    fig = px.bar(

        supplier_summary,

        x="Recommended Supplier",

        y="Procurement Score",

        title=
        "Supplier Procurement Ranking"
    )

    fig.update_layout(

        xaxis_title=
        "Supplier",

        yaxis_title=
        "Average Procurement Score",

        hovermode=
        "x unified"
    )

    return apply_enterprise_style(
        fig
    )

# =========================================================
# SAFETY RISK CHART
# =========================================================

def safety_risk_chart(

    risk_values
):

    # =====================================================
    # VALIDATION
    # =====================================================

    if risk_values is None:

        return None

    risk_values = pd.Series(
        risk_values
    ).dropna()

    if len(risk_values) == 0:

        return None

    # =====================================================
    # RISK COUNTS
    # =====================================================

    risk_counts = (

        risk_values

        .astype(str)

        .value_counts()

        .reset_index()
    )

    risk_counts.columns = [

        "Risk Level",

        "Count"
    ]

    # =====================================================
    # SORT KNOWN RISK LEVELS
    # =====================================================

    risk_order = [

        "Low",

        "Medium",

        "High"
    ]

    if all(

        level in risk_counts[
            "Risk Level"
        ].values

        for level in risk_order

    ):

        risk_counts[
            "Risk Level"
        ] = pd.Categorical(

            risk_counts[
                "Risk Level"
            ],

            categories=risk_order,

            ordered=True
        )

        risk_counts = risk_counts.sort_values(
            "Risk Level"
        )

    # =====================================================
    # BAR CHART
    # =====================================================

    fig = px.bar(

        risk_counts,

        x="Risk Level",

        y="Count",

        text="Count",

        title="Safety Risk Distribution"
    )

    fig.update_traces(

        textposition="outside"
    )

    fig.update_layout(

        xaxis_title="Risk Level",

        yaxis_title="Number of Records",

        hovermode="x unified"
    )

    return apply_enterprise_style(
        fig
    )

# =========================================================
# MODEL COMPARISON CHART
# =========================================================

def model_comparison_chart(

    model_scores
):

    if model_scores is None:

        return None

    comparison_df = pd.DataFrame(
        model_scores
    )

    fig = px.bar(

        comparison_df,

        x="Model",

        y="Score",

        title="Model Performance Comparison"
    )

    fig.update_layout(

        xaxis_title="Models",

        yaxis_title="Performance Score"
    )

    return apply_enterprise_style(fig)

# =========================================================
# CORRELATION HEATMAP
# =========================================================

def correlation_heatmap(df):

    numeric_df = df.select_dtypes(

        include=[np.number]
    )

    if numeric_df.shape[1] < 2:

        return None

    correlation_matrix = numeric_df.corr()

    fig = px.imshow(

        correlation_matrix,

        aspect="auto",

        title="Feature Correlation Heatmap"
    )

    return apply_enterprise_style(fig)

# =========================================================
# PROCUREMENT HEALTH GAUGE
# =========================================================

def create_gauge_chart(

    value,

    title="Enterprise Score",

    max_value=100
):

    fig = go.Figure()

    fig.add_trace(

        go.Indicator(

            mode="gauge+number",

            value=float(value),

            number={

                "suffix": "",

                "valueformat": ".2f",

                "font": {

                    "size": 72,

                    "color": "#f8fafc"
                }
            },

            title={

                "text": f"<b>{title}</b>",

                "font": {

                    "size": 20,

                    "color": "#f8fafc"
                }
            },

            gauge={

                "axis": {

                    "range": [

                        0,

                        max_value
                    ],

                    "tickwidth": 1,

                    "tickcolor": "#9ca3af"
                },

                "bar": {

                    "color": "#00c853",

                    "thickness": 0.35
                },

                "bgcolor": "#111827",

                "borderwidth": 2,

                "bordercolor": "#374151",

                "steps": [

                    {

                        "range": [0, 40],

                        "color": "#7f1d1d"
                    },

                    {

                        "range": [40, 70],

                        "color": "#78350f"
                    },

                    {

                        "range": [70, 100],

                        "color": "#14532d"
                    }
                ]
            }
        )
    )

    fig.update_layout(

        template="plotly_dark",

        height=500,

        margin=dict(

            l=40,

            r=40,

            t=80,

            b=40
        ),

        paper_bgcolor="#020617",

        plot_bgcolor="#020617"
    )

    return fig