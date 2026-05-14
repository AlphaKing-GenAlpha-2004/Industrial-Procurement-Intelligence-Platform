import sys
import os

# ==========================================
# PROJECT ROOT PATH FIX
# ==========================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

# ==========================================
# IMPORTS
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np

from preprocessing.detect_schema import detect_schema
from preprocessing.clean_data import clean_data
from preprocessing.outlier_handler import remove_outliers
from preprocessing.feature_builder import build_features
from preprocessing.encode_features import encode_features

from models.train_model import train_model

from visualizations import (

    create_histogram,
    create_scatter,
    create_boxplot,

    create_correlation_heatmap,

    create_feature_importance,

    create_pie_chart,
    create_line_chart,

    create_multi_feature_dashboard,

    create_actual_vs_predicted_chart,

    create_classification_comparison_chart,
    create_confusion_matrix_chart
)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(

    page_title="Industrial AI System",

    page_icon="🏭",

    layout="wide"
)

# ==========================================
# HEADER
# ==========================================

st.title(
    "🏭 Industrial AI Decision Intelligence System"
)

st.markdown("""

Industrial AI platform for:

- Demand Forecasting
- Cost Prediction
- Procurement Intelligence
- Safety Risk Analysis

""")

# ==========================================
# CACHE PREPROCESSING
# ==========================================

@st.cache_data

def preprocess_pipeline(df):

    schema = detect_schema(df)

    df = clean_data(df)

    df = remove_outliers(df)

    df = build_features(df)

    df, encoders = encode_features(df)

    return df, schema

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_files = st.file_uploader(

    "Upload CSV or Excel Files",

    type=["csv", "xlsx"],

    accept_multiple_files=True
)

# ==========================================
# MAIN PIPELINE
# ==========================================

if uploaded_files:

    # ======================================
    # LOAD DATASETS
    # ======================================

    dataframes = []

    for uploaded_file in uploaded_files:

        if uploaded_file.name.endswith(".csv"):

            temp_df = pd.read_csv(
                uploaded_file
            )

        else:

            temp_df = pd.read_excel(
                uploaded_file
            )

        dataframes.append(temp_df)

    # ======================================
    # MERGE DATASETS
    # ======================================

    df = pd.concat(

        dataframes,

        ignore_index=True
    )

    original_df = df.copy()

    # ======================================
    # SAMPLE LARGE DATASETS
    # ======================================

    if len(original_df) > 1000:

        visualization_df = original_df.sample(

            1000,

            random_state=42
        )

    else:

        visualization_df = original_df

    # ======================================
    # KPI SECTION
    # ======================================

    st.success(
        f"{len(uploaded_files)} dataset(s) uploaded successfully!"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        original_df.shape[0]
    )

    col2.metric(
        "Columns",
        original_df.shape[1]
    )

    col3.metric(
        "Files",
        len(uploaded_files)
    )

    # ======================================
    # TABS
    # ======================================

    tab1, tab2, tab3, tab4 = st.tabs([

        "Dataset",

        "Preprocessing",

        "AI Training",

        "Analytics"
    ])

    # ======================================
    # TAB 1 — DATASET
    # ======================================

    with tab1:

        st.subheader(
            "Merged Dataset"
        )

        st.dataframe(
            original_df.head(20)
        )

        st.write(
            original_df.shape
        )

    # ======================================
    # TAB 2 — PREPROCESSING
    # ======================================

    with tab2:

        with st.spinner(
            "Running preprocessing pipeline..."
        ):

            processed_df, schema = (
                preprocess_pipeline(df)
            )

        st.success(
            "Preprocessing Completed!"
        )

        st.subheader(
            "Detected Schema"
        )

        st.json(schema)

        st.subheader(
            "Processed Dataset"
        )

        st.dataframe(
            processed_df.head(20)
        )

    # ======================================
    # TAB 3 — AI TRAINING
    # ======================================

    with tab3:

        st.subheader(
            "Industrial AI Model"
        )

        target_column = st.selectbox(

            "Select Target Column",

            processed_df.columns
        )

        # ==================================
        # TRAIN MODEL
        # ==================================

        if st.button(
            "Train AI Model"
        ):

            with st.spinner(
                "Training AI model..."
            ):

                (
                    model,
                    y_test,
                    predictions,
                    problem_type

                ) = train_model(

                    processed_df,
                    target_column
                )

            st.success(
                "Model Trained Successfully!"
            )

            # ==================================
            # PREDICTION DATASET
            # ==================================

            prediction_df = pd.DataFrame({

                "Actual":
                    y_test.values,

                "Predicted":
                    predictions
            })

            # ==================================
            # REGRESSION
            # ==================================

            if problem_type == "regression":

                mae = np.mean(
                    abs(
                        y_test - predictions
                    )
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Average Actual",
                    f"{np.mean(y_test):.2f}"
                )

                c2.metric(
                    "Average Forecast",
                    f"{np.mean(predictions):.2f}"
                )

                c3.metric(
                    "Forecast Error",
                    f"{mae:.2f}"
                )

                forecast_chart = (
                    create_actual_vs_predicted_chart(

                        y_test,

                        predictions
                    )
                )

                st.plotly_chart(

                    forecast_chart,

                    use_container_width=True
                )

            # ==================================
            # CLASSIFICATION
            # ==================================

            else:

                classification_chart = (

                    create_classification_comparison_chart(

                        y_test,

                        predictions
                    )
                )

                st.plotly_chart(

                    classification_chart,

                    use_container_width=True
                )

                confusion_chart = (
                    create_confusion_matrix_chart(

                        y_test,

                        predictions
                    )
                )

                st.plotly_chart(

                    confusion_chart,

                    use_container_width=True
                )

            # ==================================
            # FEATURE IMPORTANCE
            # ==================================

            feature_chart = (
                create_feature_importance(

                    model,

                    processed_df.drop(
                        columns=[target_column]
                    ).columns
                )
            )

            if feature_chart:

                st.plotly_chart(

                    feature_chart,

                    use_container_width=True
                )

            # ==================================
            # DOWNLOAD PREDICTIONS
            # ==================================

            prediction_csv = prediction_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(

                label="Download Predicted Dataset",

                data=prediction_csv,

                file_name="predicted_results.csv",

                mime="text/csv"
            )

    # ======================================
    # TAB 4 — ANALYTICS
    # ======================================

    with tab4:

        st.subheader(
            "Industrial Analytics"
        )

        numeric_columns = (

            visualization_df.select_dtypes(
                include=['int64', 'float64']
            ).columns.tolist()
        )

        # ==================================
        # HISTOGRAM
        # ==================================

        histogram_column = st.selectbox(

            "Distribution Analysis",

            numeric_columns
        )

        histogram = create_histogram(

            visualization_df,

            histogram_column
        )

        st.plotly_chart(

            histogram,

            use_container_width=True
        )

        # ==================================
        # PIE CHART
        # ==================================

        categorical_columns = (

            visualization_df.select_dtypes(
                include=['object']
            ).columns.tolist()
        )

        if categorical_columns:

            pie_column = st.selectbox(

                "Categorical Analysis",

                categorical_columns
            )

            pie_chart = create_pie_chart(

                visualization_df,

                pie_column
            )

            st.plotly_chart(

                pie_chart,

                use_container_width=True
            )

        # ==================================
        # LINE CHART
        # ==================================

        line_column = st.selectbox(

            "Trend Analysis",

            numeric_columns,

            key="line"
        )

        line_chart = create_line_chart(

            visualization_df,

            line_column
        )

        st.plotly_chart(

            line_chart,

            use_container_width=True
        )

        # ==================================
        # SCATTER PLOT
        # ==================================

        if len(numeric_columns) >= 2:

            x_col = st.selectbox(

                "X Axis",

                numeric_columns,

                key="x"
            )

            y_col = st.selectbox(

                "Y Axis",

                numeric_columns,

                key="y"
            )

            scatter = create_scatter(

                visualization_df,

                x_col,

                y_col
            )

            st.plotly_chart(

                scatter,

                use_container_width=True
            )

        # ==================================
        # BOXPLOT
        # ==================================

        box_col = st.selectbox(

            "Boxplot Column",

            numeric_columns,

            key="box"
        )

        boxplot = create_boxplot(

            visualization_df,

            box_col
        )

        st.plotly_chart(

            boxplot,

            use_container_width=True
        )

        # ==================================
        # HEAVY VISUALS
        # ==================================

        st.subheader(
            "Advanced Analytics"
        )

        # ----------------------------------

        if st.button(
            "Generate Multi-Feature Dashboard"
        ):

            multi_dashboard = (
                create_multi_feature_dashboard(
                    visualization_df
                )
            )

            if multi_dashboard:

                st.plotly_chart(

                    multi_dashboard,

                    use_container_width=True
                )

        # ----------------------------------

        if st.button(
            "Generate Correlation Heatmap"
        ):

            heatmap = (
                create_correlation_heatmap(
                    visualization_df
                )
            )

            st.plotly_chart(

                heatmap,

                use_container_width=True
            )

    # ======================================
    # DOWNLOAD ORIGINAL DATASET
    # ======================================

    csv = original_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        label="Download Original Dataset",

        data=csv,

        file_name="industrial_dataset.csv",

        mime="text/csv"
    )