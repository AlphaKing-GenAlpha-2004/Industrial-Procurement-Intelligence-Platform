# =========================================================
# visualizations.py
# ENTERPRISE VISUALIZATION ENGINE
# CLEAN + GENERALIZED + STABLE
# =========================================================

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# IMPORTS
# =========================================================

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# GLOBAL SETTINGS
# =========================================================

PLOT_THEME = "plotly_dark"

PLOT_HEIGHT = 450

# =========================================================
# ENTERPRISE TARGET DETECTION
# =========================================================

def detect_enterprise_targets(df):

    """
    Dynamically detect enterprise targets.
    """

    targets = {

        "demand_target": None,

        "cost_target": None,

        "safety_target": None
    }

    # =====================================================
    # KEYWORDS
    # =====================================================

    demand_keywords = [

        "demand",
        "sales",
        "forecast",
        "orders",
        "volume",
        "production"
    ]

    cost_keywords = [

        "cost",
        "expense",
        "price",
        "procurement"
    ]

    safety_keywords = [

        "risk",
        "safety",
        "hazard",
        "incident"
    ]

    # =====================================================
    # DETECTION
    # =====================================================

    for column in df.columns:

        lower_col = column.lower()

        # =================================================
        # DEMAND
        # =================================================

        if any(

            keyword in lower_col

            for keyword in demand_keywords
        ):

            if (

                targets[
                    "demand_target"
                ]

                is None
            ):

                targets[
                    "demand_target"
                ] = column

        # =================================================
        # COST
        # =================================================

        if any(

            keyword in lower_col

            for keyword in cost_keywords
        ):

            if (

                targets[
                    "cost_target"
                ]

                is None
            ):

                targets[
                    "cost_target"
                ] = column

        # =================================================
        # SAFETY
        # =================================================

        if any(

            keyword in lower_col

            for keyword in safety_keywords
        ):

            if (

                targets[
                    "safety_target"
                ]

                is None
            ):

                targets[
                    "safety_target"
                ] = column

    return targets


# =========================================================
# DATASET SUMMARY
# =========================================================

def dataset_summary(df):

    """
    Enterprise dataset summary.
    """

    try:

        summary = {

            "Rows":
            int(df.shape[0]),

            "Columns":
            int(df.shape[1]),

            "Missing Values":
            int(
                df.isnull().sum().sum()
            ),

            "Duplicate Rows":
            int(
                df.duplicated().sum()
            ),

            "Memory Usage (MB)":
            round(

                df.memory_usage(
                    deep=True
                ).sum()

                /

                (1024 * 1024),

                2
            )
        }

        return pd.DataFrame({

            "Metric":
            list(summary.keys()),

            "Value":
            list(summary.values())
        })

    except:

        return pd.DataFrame()


# =========================================================
# TARGET DISTRIBUTION
# =========================================================

def plot_target_distribution(
    df,
    target_column
):

    """
    Plot enterprise target distribution.
    """

    try:

        if (

            target_column is None

            or

            target_column not in df.columns
        ):

            return None

        # =================================================
        # NUMERIC
        # =================================================

        if pd.api.types.is_numeric_dtype(

            df[target_column]
        ):

            # =============================================
            # IGNORE LOW UNIQUE NUMERIC
            # =============================================

            if (

                df[target_column]
                .nunique()

                <= 5
            ):

                return None

            fig = px.histogram(

                df,

                x=target_column,

                nbins=40,

                title=
                f"{target_column} Distribution"
            )

        # =================================================
        # CATEGORICAL
        # =================================================

        else:

            counts = (

                df[target_column]

                .astype(str)

                .value_counts()

                .head(10)
            )

            fig = px.bar(

                x=counts.index,

                y=counts.values,

                title=
                f"{target_column} Distribution"
            )

        fig.update_layout(

            template=PLOT_THEME,

            height=PLOT_HEIGHT
        )

        return fig

    except:

        return None


# =========================================================
# TREND ANALYSIS
# =========================================================

def plot_trend_analysis(
    df,
    target_column
):

    """
    Enterprise trend visualization.
    """

    try:

        if (

            target_column is None

            or

            target_column not in df.columns
        ):

            return None

        # =================================================
        # ONLY NUMERIC
        # =================================================

        if not pd.api.types.is_numeric_dtype(

            df[target_column]
        ):

            return None

        plot_df = (
            df.reset_index()
        )

        fig = px.line(

            plot_df,

            x=plot_df.index,

            y=target_column,

            title=
            f"{target_column} Trend Analysis"
        )

        fig.update_layout(

            template=PLOT_THEME,

            height=PLOT_HEIGHT,

            xaxis_title="Records",

            yaxis_title=target_column
        )

        return fig

    except:

        return None


# =========================================================
# SAFETY DISTRIBUTION
# =========================================================

def plot_safety_distribution(
    df,
    safety_column
):

    """
    Enterprise safety distribution.
    """

    try:

        if (

            safety_column is None

            or

            safety_column not in df.columns
        ):

            return None

        counts = (

            df[safety_column]

            .astype(str)

            .value_counts()

            .head(10)
        )

        fig = px.bar(

            x=counts.index,

            y=counts.values,

            title=
            "Safety Risk Distribution"
        )

        fig.update_layout(

            template=PLOT_THEME,

            height=PLOT_HEIGHT,

            xaxis_title="Safety Classes",

            yaxis_title="Count"
        )

        return fig

    except:

        return None


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def plot_feature_importance(
    importance_df,
    top_n=10
):

    """
    Enterprise feature importance plot.
    """

    try:

        if (

            importance_df is None

            or

            importance_df.empty
        ):

            return None

        plot_df = (

            importance_df

            .head(top_n)

            .sort_values(
                by="Importance"
            )
        )

        fig = px.bar(

            plot_df,

            x="Importance",

            y="Feature",

            orientation="h",

            title=
            "Top Feature Importance"
        )

        fig.update_layout(

            template=PLOT_THEME,

            height=500
        )

        return fig

    except:

        return None


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

def plot_actual_vs_predicted(
    y_true,
    predictions,
    title="Actual vs Predicted"
):

    """
    Regression evaluation visualization.
    """

    try:

        plot_df = pd.DataFrame({

            "Actual":
            y_true,

            "Predicted":
            predictions
        })

        fig = px.scatter(

            plot_df,

            x="Actual",

            y="Predicted",

            opacity=0.7,

            title=title
        )

        # =================================================
        # IDEAL LINE
        # =================================================

        min_val = min(

            plot_df["Actual"].min(),

            plot_df["Predicted"].min()
        )

        max_val = max(

            plot_df["Actual"].max(),

            plot_df["Predicted"].max()
        )

        fig.add_trace(

            go.Scatter(

                x=[min_val, max_val],

                y=[min_val, max_val],

                mode="lines",

                name="Ideal"
            )
        )

        fig.update_layout(

            template=PLOT_THEME,

            height=500
        )

        return fig

    except:

        return None


# =========================================================
# CLASSIFICATION DISTRIBUTION
# =========================================================

def plot_classification_distribution(
    predictions,
    title="Classification Distribution"
):

    """
    Classification prediction distribution.
    """

    try:

        counts = (

            pd.Series(predictions)

            .astype(str)

            .value_counts()
        )

        fig = px.bar(

            x=counts.index,

            y=counts.values,

            title=title
        )

        fig.update_layout(

            template=PLOT_THEME,

            height=PLOT_HEIGHT,

            xaxis_title="Classes",

            yaxis_title="Count"
        )

        return fig

    except:

        return None


# =========================================================
# MODEL SUMMARY TABLE
# =========================================================

def model_summary_table(
    model_output
):

    """
    Enterprise model summary table.
    """

    try:

        summary = {

            "Target Column":

            model_output.get(
                "target_column",
                "Unknown"
            ),

            "Problem Type":

            model_output.get(
                "problem_type",
                "Unknown"
            )
        }

        metrics = model_output.get(
            "metrics",
            {}
        )

        for key, value in metrics.items():

            summary[key] = value

        return pd.DataFrame({

            "Metric":
            list(summary.keys()),

            "Value":
            list(summary.values())
        })

    except:

        return pd.DataFrame()


# =========================================================
# ENTERPRISE PREDICTION TABLE
# =========================================================

def enterprise_prediction_table(
    prediction_results
):

    """
    Enterprise prediction summary.
    """

    try:

        return pd.DataFrame({

            "Prediction":
            list(
                prediction_results.keys()
            ),

            "Value":
            list(
                prediction_results.values()
            )
        })

    except:

        return pd.DataFrame()


# =========================================================
# ENTERPRISE ALERT TABLE
# =========================================================

def enterprise_alert_table(
    alerts
):

    """
    Enterprise alerts summary.
    """

    try:

        if not isinstance(alerts, list):

            alerts = [alerts]

        return pd.DataFrame({

            "Enterprise Alerts":
            alerts
        })

    except:

        return pd.DataFrame()


# =========================================================
# ENTERPRISE REPORT TABLE
# =========================================================

def enterprise_report_table(
    report_df
):

    """
    Enterprise report preview.
    """

    try:

        if report_df is None:

            return pd.DataFrame()

        return report_df

    except:

        return pd.DataFrame()


# =========================================================
# GENERATE ENTERPRISE VISUALS
# =========================================================

def generate_enterprise_visuals(df):

    """
    Generate clean enterprise visuals.
    """

    visuals = {}

    try:

        targets = detect_enterprise_targets(
            df
        )

        demand_target = (
            targets["demand_target"]
        )

        cost_target = (
            targets["cost_target"]
        )

        safety_target = (
            targets["safety_target"]
        )

        # =================================================
        # DEMAND VISUALS
        # =================================================

        if demand_target is not None:

            visuals[
                "demand_distribution"
            ] = plot_target_distribution(

                df,

                demand_target
            )

            visuals[
                "demand_trend"
            ] = plot_trend_analysis(

                df,

                demand_target
            )

        # =================================================
        # COST VISUALS
        # =================================================

        if cost_target is not None:

            visuals[
                "cost_distribution"
            ] = plot_target_distribution(

                df,

                cost_target
            )

        # =================================================
        # SAFETY VISUALS
        # =================================================

        if safety_target is not None:

            visuals[
                "safety_distribution"
            ] = plot_safety_distribution(

                df,

                safety_target
            )

    except Exception as e:

        print(
            f"Visualization Error: {e}"
        )

    return visuals