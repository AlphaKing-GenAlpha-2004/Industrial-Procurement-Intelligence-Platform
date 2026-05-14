import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

import pandas as pd


# ==========================================
# HISTOGRAM
# ==========================================

def create_histogram(df, column):

    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        title=f"{column} Distribution",
        template="plotly_dark"
    )

    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Frequency",
        height=500
    )

    return fig


# ==========================================
# SCATTER PLOT
# ==========================================

def create_scatter(df, x_col, y_col):

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=y_col,
        title=f"{x_col} vs {y_col}",
        template="plotly_dark"
    )

    fig.update_layout(
        height=500
    )

    return fig


# ==========================================
# BOXPLOT
# ==========================================

def create_boxplot(df, column):

    fig = px.box(
        df,
        y=column,
        title=f"{column} Boxplot",
        template="plotly_dark"
    )

    fig.update_layout(
        height=500
    )

    return fig


# ==========================================
# CORRELATION HEATMAP
# ==========================================

def create_correlation_heatmap(df):

    numeric_df = df.select_dtypes(
        include=['int64', 'float64']
    )

    correlation = numeric_df.corr()

    fig = px.imshow(
        correlation,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Feature Correlation Heatmap"
    )

    fig.update_layout(
        height=700
    )

    return fig


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

def create_feature_importance(
    model,
    feature_names
):

    if not hasattr(
        model,
        "feature_importances_"
    ):

        return None

    importance_df = pd.DataFrame({

        "Feature":
            feature_names,

        "Importance":
            model.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
    )

    fig = px.bar(
        importance_df.head(15),
        x="Importance",
        y="Feature",
        orientation='h',
        title="Top Feature Importance",
        template="plotly_dark"
    )

    fig.update_layout(
        height=600
    )

    return fig


# ==========================================
# PIE CHART
# ==========================================

def create_pie_chart(df, column):

    value_counts = (
        df[column]
        .value_counts()
        .reset_index()
    )

    value_counts.columns = [
        column,
        "Count"
    ]

    fig = px.pie(
        value_counts,
        names=column,
        values="Count",
        title=f"{column} Distribution",
        template="plotly_dark"
    )

    fig.update_layout(
        height=500
    )

    return fig


# ==========================================
# LINE CHART
# ==========================================

def create_line_chart(df, column):

    # ======================================
    # SORT VALUES
    # ======================================

    sorted_df = df.sort_values(
        by=column
    ).reset_index(drop=True)

    # ======================================
    # SMOOTHING
    # ======================================

    sorted_df["Smoothed"] = (

        sorted_df[column]
        .rolling(
            window=25,
            min_periods=1
        )
        .mean()
    )

    # ======================================
    # CREATE FIGURE
    # ======================================

    fig = px.line(

        sorted_df,

        y="Smoothed",

        title=f"{column} Trend Analysis",

        template="plotly_dark"
    )

    fig.update_layout(

        xaxis_title="Observations",

        yaxis_title=column,

        height=500
    )

    return fig


# ==========================================
# MULTI FEATURE DASHBOARD
# ==========================================

def create_multi_feature_dashboard(df):

    numeric_columns = df.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    if len(numeric_columns) < 4:

        return None

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            numeric_columns[:4]
        )
    )

    for i, col in enumerate(
        numeric_columns[:4]
    ):

        row = (i // 2) + 1
        col_position = (i % 2) + 1

        fig.add_trace(

            go.Histogram(
                x=df[col],
                name=col
            ),

            row=row,
            col=col_position
        )

    fig.update_layout(
        height=800,
        title="Industrial AI Feature Dashboard",
        template="plotly_dark"
    )

    return fig
    
# ==========================================
# INDUSTRIAL FORECAST COMPARISON
# ==========================================

def create_actual_vs_predicted_chart(
    y_test,
    predictions
):

    comparison_df = pd.DataFrame({

        "Actual":
            y_test.values,

        "Predicted":
            predictions
    })

    # ==================================
    # REDUCE VISUAL NOISE
    # ==================================

    # Sort values
    comparison_df = (
        comparison_df
        .reset_index(drop=True)
    )

    # Rolling averages
    comparison_df["Actual_Smoothed"] = (
        comparison_df["Actual"]
        .rolling(window=25)
        .mean()
    )

    comparison_df["Predicted_Smoothed"] = (
        comparison_df["Predicted"]
        .rolling(window=25)
        .mean()
    )

    # ==================================
    # CREATE GRAPH
    # ==================================

    fig = go.Figure()

    # Actual trend
    fig.add_trace(

        go.Scatter(

            y=comparison_df[
                "Actual_Smoothed"
            ],

            mode='lines',

            name='Actual Trend',

            line=dict(width=4)
        )
    )

    # Predicted trend
    fig.add_trace(

        go.Scatter(

            y=comparison_df[
                "Predicted_Smoothed"
            ],

            mode='lines',

            name='Predicted Trend',

            line=dict(width=4)
        )
    )

    fig.update_layout(

        title=(
            "Demand Forecast Trend Comparison"
        ),

        xaxis_title="Production Timeline",

        yaxis_title="Demand / Cost",

        template="plotly_dark",

        height=600,

        legend=dict(
            orientation="h"
        )
    )

    return fig

# ==========================================
# ACTUAL VS PREDICTED CLASSIFICATION
# ==========================================

def create_classification_comparison_chart(
    y_test,
    predictions
):

    comparison_df = pd.DataFrame({

        "Actual":
            y_test.values,

        "Predicted":
            predictions
    })

    fig = go.Figure()

    # Actual labels
    fig.add_trace(

        go.Scatter(

            y=comparison_df["Actual"],

            mode='markers',

            name='Actual',

            marker=dict(size=10)
        )
    )

    # Predicted labels
    fig.add_trace(

        go.Scatter(

            y=comparison_df["Predicted"],

            mode='markers',

            name='Predicted',

            marker=dict(size=10)
        )
    )

    fig.update_layout(

        title=(
            "Actual vs Predicted Classes"
        ),

        xaxis_title="Samples",

        yaxis_title="Class Labels",

        template="plotly_dark",

        height=600
    )

    return fig

from sklearn.metrics import confusion_matrix
import numpy as np

# ==========================================
# CONFUSION MATRIX
# ==========================================

def create_confusion_matrix_chart(
    y_test,
    predictions
):

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fig = px.imshow(

        cm,

        text_auto=True,

        color_continuous_scale="Blues",

        title="Confusion Matrix"
    )

    fig.update_layout(
        height=600
    )

    return fig