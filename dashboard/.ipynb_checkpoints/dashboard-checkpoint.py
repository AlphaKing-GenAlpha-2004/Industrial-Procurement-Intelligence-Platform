# =========================================================
# dashboard.py
# FINAL GENERALIZABLE ENTERPRISE PROCUREMENT DASHBOARD
# CLEAN + STABLE + FAST + PROCUREMENT-FOCUSED
# =========================================================

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# IMPORTS
# =========================================================

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import traceback

# =========================================================
# ROOT PATH
# =========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ROOT_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..")
)

if ROOT_DIR not in sys.path:

    sys.path.append(ROOT_DIR)

# =========================================================
# LOCAL IMPORTS
# =========================================================

from preprocessing.data_processor import (
    preprocess_dataset,
    detect_task_types
)

from preprocessing.validation import (
    validate_dataset
)

from models.model_engine import (
    run_enterprise_pipeline
)

from visualization.plotly_charts import (

    demand_prediction_chart,

    cost_prediction_chart,

    feature_importance_chart,

    procurement_score_chart,

    supplier_ranking_chart,

    safety_risk_chart
)

from navigation import (

    enterprise_navigation,

    page_header,

    section_title,

    section_divider,

    success_banner,

    error_banner,

    empty_state
)

from theme import (

    load_enterprise_theme,

    hero_header
)

# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(

    page_title="Enterprise Procurement Intelligence",

    page_icon="📊",

    layout="wide",

    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD THEME
# =========================================================

load_enterprise_theme()

hero_header()

# =========================================================
# NAVIGATION
# =========================================================

selected_page = enterprise_navigation()

# =========================================================
# RESET
# =========================================================

if st.sidebar.button(
    "Reset Analysis"
):

    st.session_state.clear()

    st.rerun()

# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_files = st.sidebar.file_uploader(

    "Upload Enterprise Datasets",

    type=["csv", "xlsx"],

    accept_multiple_files=True
)

# =========================================================
# EMPTY STATE
# =========================================================

if not uploaded_files:

    empty_state(
        "Upload one or more enterprise datasets."
    )

    st.stop()

# =========================================================
# DATASET SIGNATURE
# =========================================================

current_signature = sorted(

    file.name

    for file in uploaded_files
)

if (

    st.session_state.get(
        "uploaded_dataset_signature"
    )

    !=

    current_signature

):

    st.session_state[
        "uploaded_dataset_signature"
    ] = current_signature

    st.session_state[
        "enterprise_cache"
    ] = None

# =========================================================
# LOAD DATASETS
# =========================================================

datasets = {}

for file in uploaded_files:

    try:

        if file.name.lower().endswith(".csv"):

            df = pd.read_csv(
                file
            )

        else:

            df = pd.read_excel(
                file
            )

        if df.empty:

            st.warning(
                f"{file.name} is empty."
            )

            continue

        datasets[file.name] = df

        print("\n================================")
        print(f"DATASET LOADED : {file.name}")
        print(f"ROWS           : {len(df)}")
        print(f"COLUMNS        : {len(df.columns)}")
        print("================================")

    except Exception as e:

        st.error(
            f"Failed to load {file.name}: {e}"
        )

        st.stop()

# =========================================================
# VALIDATION
# =========================================================

if not datasets:

    st.error(
        "No valid datasets uploaded."
    )

    st.stop()

# =========================================================
# SINGLE DATASET MODE
# =========================================================

if len(datasets) == 1:

    dataset_name = next(
        iter(datasets)
    )

    raw_df = datasets[
        dataset_name
    ].copy()

    print("\n================================")
    print("SINGLE DATASET MODE")
    print("================================")
    print(f"DATASET : {dataset_name}")
    print(f"ROWS    : {len(raw_df)}")
    print(f"COLUMNS : {len(raw_df.columns)}")
    print("================================")

# =========================================================
# MULTI DATASET MODE
# =========================================================

else:

    common_columns = None

    for df in datasets.values():

        cols = set(df.columns)

        if common_columns is None:

            common_columns = cols

        else:

            common_columns &= cols

    common_columns = sorted(
        list(common_columns)
    )

    if not common_columns:

        st.error(

            """
No common columns found.

Datasets cannot be merged.
"""
        )

        st.stop()

    merge_key = st.sidebar.selectbox(

        "Select Merge Key",

        common_columns
    )

    # =====================================================
    # CACHE INVALIDATION
    # =====================================================

    if (

        st.session_state.get(
            "selected_merge_key"
        )

        !=

        merge_key

    ):

        st.session_state[
            "selected_merge_key"
        ] = merge_key

        st.session_state[
            "enterprise_cache"
        ] = None

    # =====================================================
    # MERGE FUNCTION
    # =====================================================

    def merge_enterprise_datasets(

        datasets,

        merge_key

    ):

        merged_df = None

        for dataset_name, df in datasets.items():

            if merge_key not in df.columns:

                continue

            # =========================================
            # PREVENT MERGE EXPLOSION
            # =========================================

            df = df.drop_duplicates(
                subset=[merge_key]
            )

            print("\n================================")
            print(f"DATASET : {dataset_name}")
            print(f"ROWS    : {len(df)}")
            print(
                f"UNIQUE KEYS : "
                f"{df[merge_key].nunique()}"
            )
            print("================================")

            if merged_df is None:

                merged_df = df.copy()

            else:

                merged_df = pd.merge(

                    merged_df,

                    df,

                    on=merge_key,

                    how="outer",

                    suffixes=(

                        "",

                        f"_{dataset_name}"
                    )
                )

        if merged_df is None:

            return pd.DataFrame()

        merged_df = merged_df.drop_duplicates()

        print("\n================================")
        print("MERGE COMPLETE")
        print("================================")
        print(f"ROWS    : {len(merged_df)}")
        print(f"COLUMNS : {len(merged_df.columns)}")
        print("================================")

        return merged_df

    raw_df = merge_enterprise_datasets(

        datasets,

        merge_key
    )

# =========================================================
# FINAL VALIDATION
# =========================================================

if raw_df.empty:

    st.error(
        "Final dataset is empty."
    )

    st.stop()

# =========================================================
# ROW BASELINE
# =========================================================

st.session_state[
    "original_dataset_rows"
] = len(raw_df)

# =========================================================
# FINAL DIAGNOSTICS
# =========================================================

print("\n================================")
print("FINAL DATASET SUMMARY")
print("================================")

print(
    f"ROWS : {len(raw_df)}"
)

print(
    f"COLUMNS : {len(raw_df.columns)}"
)

print(
    f"DUPLICATES : "
    f"{raw_df.duplicated().sum()}"
)

print(
    f"MISSING VALUES : "
    f"{raw_df.isnull().sum().sum()}"
)

print("================================")

# =========================================================
# SESSION CACHE
# =========================================================

if "enterprise_cache" not in st.session_state:

    st.session_state[
        "enterprise_cache"
    ] = None

# =========================================================
# RUN PIPELINE
# =========================================================

if st.session_state["enterprise_cache"] is None:

    with st.spinner(

        "Running Enterprise AI Pipeline..."

    ):

        try:

            # =========================================
            # DATASET VALIDATION
            # =========================================

            validation_report = (

                validate_dataset(
                    raw_df
                )
            )

            # =========================================
            # PREPROCESSING
            # =========================================

            preprocessing_output = (

                preprocess_dataset(
                    raw_df
                )
            )

            # =========================================
            # EXTRACT OBJECTS
            # =========================================

            processed_df = (

                preprocessing_output[
                    "processed_df"
                ]
            )

            raw_business_df = (

                preprocessing_output[
                    "raw_business_df"
                ]
            )

            targets = (

                preprocessing_output[
                    "targets"
                ]
            )

            task_types = (

                preprocessing_output[
                    "task_types"
                ]
            )

            feature_columns = (

                preprocessing_output[
                    "feature_columns"
                ]
            )

            target_columns = (

                preprocessing_output.get(

                    "target_columns",

                    []
                )
            )

            label_encoders = (

                preprocessing_output.get(

                    "label_encoders",

                    {}
                )
            )

            diagnostics = (

                preprocessing_output.get(

                    "diagnostics",

                    {}
                )
            )

            entity_column = (

                preprocessing_output.get(

                    "entity_column",

                    None
                )
            )

            supplier_column = (

                preprocessing_output.get(

                    "supplier_column",

                    None
                )
            )

            target_confidence = (

                preprocessing_output.get(

                    "target_confidence",

                    {}
                )
            )

            target_candidates = (

                preprocessing_output.get(

                    "target_candidates",

                    {}
                )
            )

            # =========================================
            # VALIDATION
            # =========================================

            if processed_df.empty:

                raise ValueError(

                    "Processed dataframe is empty"

                )

            if raw_business_df.empty:

                raise ValueError(

                    "Business dataframe is empty"

                )

            # =========================================
            # MODEL PIPELINE
            # =========================================

            pipeline_output = (

                run_enterprise_pipeline(

                    processed_df=
                    processed_df,

                    raw_business_df=
                    raw_business_df,

                    targets=
                    targets,

                    task_types=
                    task_types,

                    feature_columns=
                    feature_columns,

                    entity_column=
                    entity_column,

                    supplier_column=
                    supplier_column
                )
            )

            # =========================================
            # CACHE RESULTS
            # =========================================

            st.session_state[
                "enterprise_cache"
            ] = {

                # -------------------------------------
                # DATA VALIDATION
                # -------------------------------------

                "validation_report":
                validation_report,

                # -------------------------------------
                # DATASETS
                # -------------------------------------

                "processed_df":
                processed_df,

                "raw_business_df":
                raw_business_df,

                # -------------------------------------
                # TARGETS
                # -------------------------------------

                "targets":
                targets,

                "task_types":
                task_types,

                "target_confidence":
                target_confidence,

                "target_candidates":
                target_candidates,

                # -------------------------------------
                # FEATURES
                # -------------------------------------

                "feature_columns":
                feature_columns,

                "target_columns":
                target_columns,

                "label_encoders":
                label_encoders,

                # -------------------------------------
                # PROCUREMENT INTELLIGENCE
                # -------------------------------------

                "entity_column":
                entity_column,

                "supplier_column":
                supplier_column,

                # -------------------------------------
                # DIAGNOSTICS
                # -------------------------------------

                "preprocessing_diagnostics":
                diagnostics,

                # -------------------------------------
                # MODEL OUTPUTS
                # -------------------------------------

                "pipeline_output":
                pipeline_output
            }

            print("\n================================")
            print("PIPELINE COMPLETED")
            print("================================")
            print(
                f"Rows: {len(raw_business_df)}"
            )
            print(
                f"Features: {len(feature_columns)}"
            )
            print(
                f"Entity Column: {entity_column}"
            )
            print(
                f"Supplier Column: {supplier_column}"
            )
            print(
                f"Targets: {targets}"
            )
            print("================================\n")
        
        except Exception as e:
            st.error(f"Pipeline Failed: {str(e)}")
            st.code(traceback.format_exc())
            print(traceback.format_exc())
            
            st.stop()

# =========================================================
# LOAD CACHE
# =========================================================

cache = st.session_state.get(
    "enterprise_cache"
)

if cache is None:

    cache = {}

# =========================================================
# DATA OBJECTS
# =========================================================

validation_report = cache.get(

    "validation_report",

    {}
)

processed_df = cache.get(

    "processed_df",

    pd.DataFrame()
)

raw_business_df = cache.get(

    "raw_business_df",

    pd.DataFrame()
)

# =========================================================
# TARGET OBJECTS
# =========================================================

targets = cache.get(

    "targets",

    {}
)

task_types = cache.get(

    "task_types",

    {}
)

target_confidence = cache.get(

    "target_confidence",

    {}
)

target_candidates = cache.get(

    "target_candidates",

    {}
)

# =========================================================
# FEATURE OBJECTS
# =========================================================

feature_columns = cache.get(

    "feature_columns",

    []
)

target_columns = cache.get(

    "target_columns",

    []
)

label_encoders = cache.get(

    "label_encoders",

    {}
)

# =========================================================
# PROCUREMENT OBJECTS
# =========================================================

entity_column = cache.get(

    "entity_column",

    None
)

supplier_column = cache.get(

    "supplier_column",

    None
)

# =========================================================
# DIAGNOSTICS
# =========================================================

preprocessing_diagnostics = cache.get(

    "preprocessing_diagnostics",

    {}
)

pipeline_output = cache.get(

    "pipeline_output",

    {}
)

pipeline_diagnostics = (

    pipeline_output.get(

        "diagnostics",

        {}
    )
)

pipeline_errors = (

    pipeline_output.get(

        "errors",

        {}
    )
)

# =========================================================
# SUPPLIER INTELLIGENCE
# =========================================================

supplier_intelligence_df = (

    pipeline_output.get(

        "supplier_intelligence_df",

        pd.DataFrame()
    )
)

print("\n========== SUPPLIER DF ==========")

print(
    "Type:",
    type(supplier_intelligence_df)
)

if supplier_intelligence_df is not None:

    print(
        "Shape:",
        supplier_intelligence_df.shape
    )

    print(
        "Empty:",
        supplier_intelligence_df.empty
    )

    if not supplier_intelligence_df.empty:

        print(
            supplier_intelligence_df.head()
        )

else:

    print(
        "supplier_intelligence_df = None"
    )

print("=================================\n")

# =========================================================
# MODEL OBJECTS
# =========================================================

demand_model = (

    pipeline_output.get(

        "demand_model",

        None
    )
)

cost_model = (

    pipeline_output.get(

        "cost_model",

        None
    )
)

safety_model = (

    pipeline_output.get(

        "safety_model",

        None
    )
)

# =========================================================
# CACHE VALIDATION
# =========================================================

if cache is None:

    st.error(

        "Enterprise cache unavailable."

    )

    st.stop()

if processed_df.empty:

    st.error(

        """
No processed dataset found.

Please upload and process
a dataset first.
"""
    )

    st.stop()

if raw_business_df.empty:

    st.error(

        """
No business dataset found.

Please rerun preprocessing.
"""
    )

    st.stop()

# =========================================================
# TARGET VALIDATION
# =========================================================

required_targets = [

    "demand_target",

    "cost_target",

    "safety_target"
]

for target_key in required_targets:

    if target_key not in targets:

        st.error(

            f"Missing target: {target_key}"

        )

        st.stop()

# =========================================================
# SIDEBAR DIAGNOSTICS
# =========================================================

with st.sidebar.expander(

    "Pipeline Diagnostics",

    expanded=False

):

    st.json(
        pipeline_diagnostics
    )

with st.sidebar.expander(

    "Preprocessing Diagnostics",

    expanded=False

):

    st.json(
        preprocessing_diagnostics
    )

with st.sidebar.expander(

    "Target Detection Confidence",

    expanded=False

):

    st.json(
        target_confidence
    )

# =========================================================
# PROCUREMENT METADATA
# =========================================================

with st.sidebar.expander(

    "Entity Detection",

    expanded=False

):

    st.write(

        "Entity Column:",

        entity_column
    )

    st.write(

        "Supplier Column:",

        supplier_column
    )

# =========================================================
# TARGET CONFIGURATION
# =========================================================

from preprocessing.data_processor import (
    detect_task_types
)

st.sidebar.subheader(

    "Target Configuration"
)

available_columns = list(

    raw_business_df.columns
)

# =========================================================
# DEBUG
# =========================================================

print("\n================================")
print("CACHE LOADED")
print("================================")

print(

    f"Rows: {len(raw_business_df)}"
)

print(

    f"Features: {len(feature_columns)}"
)

print(

    f"Entity Column: {entity_column}"
)

print(

    f"Supplier Column: {supplier_column}"
)

print(

    f"Targets: {targets}"
)

print(

    f"Confidence: {target_confidence}"
)

print("================================\n")

if (
    "supplier_intelligence_df"
    in
    pipeline_output
):

    print(
        "CACHE DF SHAPE:",
        pipeline_output[
            "supplier_intelligence_df"
        ].shape
    )

else:

    print(
        "supplier_intelligence_df missing from cache"
    )

# =========================================================
# DEMAND TARGET
# =========================================================

default_demand = 0

if (

    targets.get(
        "demand_target"
    )

    in available_columns

):

    default_demand = (

        available_columns.index(

            targets.get(
                "demand_target"
            )
        )
    )

selected_demand_target = (

    st.sidebar.selectbox(

        "Demand Target",

        available_columns,

        index=default_demand
    )
)

# =========================================================
# COST TARGET
# =========================================================

default_cost = 0

if (

    targets.get(
        "cost_target"
    )

    in available_columns

):

    default_cost = (

        available_columns.index(

            targets.get(
                "cost_target"
            )
        )
    )

selected_cost_target = (

    st.sidebar.selectbox(

        "Cost Target",

        available_columns,

        index=default_cost
    )
)

# =========================================================
# SAFETY TARGET
# =========================================================

default_safety = 0

if (

    targets.get(
        "safety_target"
    )

    in available_columns

):

    default_safety = (

        available_columns.index(

            targets.get(
                "safety_target"
            )
        )
    )

selected_safety_target = (

    st.sidebar.selectbox(

        "Safety Target",

        available_columns,

        index=default_safety
    )
)

# =========================================================
# INHOUSE COST TARGET
# =========================================================

default_inhouse_cost = 0

if (

    targets.get(
        "inhouse_cost_target"
    )

    in available_columns

):

    default_inhouse_cost = (

        available_columns.index(

            targets.get(
                "inhouse_cost_target"
            )
        )
    )

selected_inhouse_cost_target = (

    st.sidebar.selectbox(

        "In-House Cost Target",

        available_columns,

        index=default_inhouse_cost
    )
)

# =========================================================
# INHOUSE CAPACITY TARGET
# =========================================================

default_inhouse_capacity = 0

if (

    targets.get(
        "inhouse_capacity_target"
    )

    in available_columns

):

    default_inhouse_capacity = (

        available_columns.index(

            targets.get(
                "inhouse_capacity_target"
            )
        )
    )

selected_inhouse_capacity_target = (

    st.sidebar.selectbox(

        "In-House Capacity Target",

        available_columns,

        index=default_inhouse_capacity
    )
)

# =========================================================
# INHOUSE SAFETY TARGET
# =========================================================

default_inhouse_safety = 0

if (

    targets.get(
        "inhouse_safety_target"
    )

    in available_columns

):

    default_inhouse_safety = (

        available_columns.index(

            targets.get(
                "inhouse_safety_target"
            )
        )
    )

selected_inhouse_safety_target = (

    st.sidebar.selectbox(

        "In-House Safety Target",

        available_columns,

        index=default_inhouse_safety
    )
)

# =========================================================
# ENTITY COLUMN
# =========================================================

st.sidebar.subheader(
    "Entity Intelligence"
)

entity_options = [

    "None"

] + available_columns

entity_default = 0

if (

    entity_column

    in available_columns

):

    entity_default = (

        entity_options.index(
            entity_column
        )
    )

selected_entity_column = (

    st.sidebar.selectbox(

        "Part / Equipment Column",

        entity_options,

        index=entity_default
    )
)

# =========================================================
# SUPPLIER COLUMN
# =========================================================

supplier_default = 0

if (

    supplier_column

    in available_columns

):

    supplier_default = (

        entity_options.index(
            supplier_column
        )
    )

selected_supplier_column = (

    st.sidebar.selectbox(

        "Supplier Column",

        entity_options,

        index=supplier_default
    )
)

# =========================================================
# VALIDATION
# =========================================================

selected_targets = [

    selected_demand_target,

    selected_cost_target,

    selected_safety_target,

    selected_inhouse_cost_target,

    selected_inhouse_capacity_target,

    selected_inhouse_safety_target
]

selected_targets = [

    x

    for x in selected_targets

    if x is not None
]

if len(

    selected_targets

) != len(

    set(selected_targets)

):

    st.sidebar.error(

        "Demand, Cost and Safety targets must be different."
    )

    st.stop()

# =========================================================
# CHANGE DETECTION
# =========================================================

targets_changed = (

    selected_demand_target
    !=
    targets.get("demand_target")

    or

    selected_cost_target
    !=
    targets.get("cost_target")

    or

    selected_safety_target
    !=
    targets.get("safety_target")

    or

    selected_inhouse_cost_target
    !=
    targets.get("inhouse_cost_target")

    or

    selected_inhouse_capacity_target
    !=
    targets.get("inhouse_capacity_target")

    or

    selected_inhouse_safety_target
    !=
    targets.get("inhouse_safety_target")

    or

    (
        selected_entity_column
        if selected_entity_column != "None"
        else None
    )
    !=
    entity_column

    or

    (
        selected_supplier_column
        if selected_supplier_column != "None"
        else None
    )
    !=
    supplier_column
)

# =========================================================
# RETRAIN
# =========================================================

if targets_changed:

    targets = {

        "demand_target":
        selected_demand_target,

        "cost_target":
        selected_cost_target,

        "safety_target":
        selected_safety_target,

        "inhouse_cost_target":
        selected_inhouse_cost_target,

        "inhouse_capacity_target":
        selected_inhouse_capacity_target,

        "inhouse_safety_target":
        selected_inhouse_safety_target
    }

    task_types = (

        detect_task_types(

            raw_business_df,

            targets
        )
    )

    entity_column = (

        selected_entity_column

        if

        selected_entity_column != "None"

        else None
    )

    supplier_column = (

        selected_supplier_column

        if

        selected_supplier_column != "None"

        else None
    )

    with st.spinner(

        "Retraining models..."
    ):
        # Clear old cache before retraining
        
        st.session_state["enterprise_cache"] = {}

        pipeline_output = (

            run_enterprise_pipeline(

                processed_df=
                processed_df,

                raw_business_df=
                raw_business_df,

                targets=
                targets,

                task_types=
                task_types,

                feature_columns=
                feature_columns,

                entity_column=
                entity_column,

                supplier_column=
                supplier_column
            )
        )
        print("\nAFTER RETRAIN ->",pipeline_output.get("supplier_intelligence_df",pd.DataFrame()).shape)

    st.session_state[
        "enterprise_cache"
    ].update({

        "targets":
        targets,

        "task_types":
        task_types,

        "entity_column":
        entity_column,

        "supplier_column":
        supplier_column,

        "pipeline_output":
        pipeline_output
    })

    print("\nCACHE STORED ->",st.session_state["enterprise_cache"]["pipeline_output"]
          .get("supplier_intelligence_df",pd.DataFrame()).shape)

    st.rerun()

# =========================================================
# ACTIVE CONFIGURATION
# =========================================================

st.sidebar.success(

    f"""
Demand : {targets.get('demand_target')}

Cost : {targets.get('cost_target')}

Safety : {targets.get('safety_target')}

In-House Cost :
{targets.get('inhouse_cost_target')}

In-House Capacity :
{targets.get('inhouse_capacity_target')}

In-House Safety :
{targets.get('inhouse_safety_target')}

Entity : {entity_column}

Supplier : {supplier_column}
"""
)

# =========================================================
# GENERATED MODELS
# =========================================================

generated_models = [

    key

    for key

    in pipeline_output.keys()

    if key.endswith(
        "_model"
    )
]

if len(
    generated_models
) > 0:

    success_banner(

        f"{len(generated_models)} Models Trained Successfully"

    )

else:

    error_banner(

        "No Models Were Successfully Trained"

    )

# =========================================================
# MODEL OUTPUTS
# =========================================================

demand_model = (

    pipeline_output.get(
        "demand_model"
    )
)

cost_model = (

    pipeline_output.get(
        "cost_model"
    )
)

safety_model = (

    pipeline_output.get(
        "safety_model"
    )
)

supplier_intelligence_df = (

    pipeline_output.get(

        "supplier_intelligence_df",

        pd.DataFrame()
    )
)

print("\n========== SUPPLIER DF ==========")

print(
    "Type:",
    type(
        supplier_intelligence_df
    )
)

print(
    "Shape:",
    supplier_intelligence_df.shape
)

print(
    "Empty:",
    supplier_intelligence_df.empty
)

print("=================================\n")

# =========================================================
# PIPELINE ERRORS
# =========================================================

pipeline_errors = (

    pipeline_output.get(

        "errors",

        {}
    )
)

with st.sidebar.expander(

    "Pipeline Errors",

    expanded=False

):

    if pipeline_errors:

        st.json(
            pipeline_errors
        )

    else:

        st.success(
            "No pipeline errors detected."
        )

# =========================================================
# ENTERPRISE DATAFRAME
# =========================================================

enterprise_df = (

    raw_business_df.copy()

    .reset_index(
        drop=True
    )
)

TOTAL_ROWS = len(
    enterprise_df
)

# =========================================================
# OUTPUT VALIDATION
# =========================================================

model_mapping = {

    "Demand":
    demand_model,

    "Cost":
    cost_model,

    "Safety":
    safety_model
}

validation_errors = []

for model_name, model_output in model_mapping.items():

    if model_output is None:

        continue

    try:

        prediction_count = len(

            model_output[
                "full_predictions"
            ]
        )

        actual_count = len(

            model_output[
                "full_actual"
            ]
        )

        if prediction_count != TOTAL_ROWS:

            validation_errors.append(

                f"{model_name} Predictions "
                f"({prediction_count}) "
                f"!= Dataset Rows "
                f"({TOTAL_ROWS})"
            )

        if actual_count != TOTAL_ROWS:

            validation_errors.append(

                f"{model_name} Actuals "
                f"({actual_count}) "
                f"!= Dataset Rows "
                f"({TOTAL_ROWS})"
            )

    except Exception as e:

        validation_errors.append(

            f"{model_name} Validation Error -> {e}"
        )

# =========================================================
# VALIDATION FAILURE
# =========================================================

if len(validation_errors) > 0:

    st.error(

        "Model Output Validation Failed"
    )

    for error in validation_errors:

        st.error(error)

    st.stop()

# =========================================================
# OUTPUT DIAGNOSTICS
# =========================================================

with st.sidebar.expander(

    "Output Diagnostics",

    expanded=False

):

    st.write(
        "Rows:",
        TOTAL_ROWS
    )

    st.write(
        "Demand Model:",
        demand_model is not None
    )

    st.write(
        "Cost Model:",
        cost_model is not None
    )

    st.write(
        "Safety Model:",
        safety_model is not None
    )



# =========================================================
# DEMAND OUTPUTS
# =========================================================

if demand_model is not None:

    try:

        enterprise_df[
            "Actual Demand"
        ] = demand_model[
            "full_actual"
        ]

        enterprise_df[
            "Predicted Demand"
        ] = np.round(

            demand_model[
                "full_predictions"
            ],

            2
        )

        if (

            task_types.get(
                "demand_target"
            )

            ==

            "regression"
        ):

            enterprise_df[
                "Demand Error"
            ] = (

                pd.to_numeric(

                    enterprise_df[
                        "Actual Demand"
                    ],

                    errors="coerce"
                )

                -

                pd.to_numeric(

                    enterprise_df[
                        "Predicted Demand"
                    ],

                    errors="coerce"
                )
            )

    except Exception as e:

        print(
            f"Demand Output Error -> {e}"
        )

# =========================================================
# COST OUTPUTS
# =========================================================

if cost_model is not None:

    try:

        enterprise_df[
            "Actual Cost"
        ] = cost_model[
            "full_actual"
        ]

        enterprise_df[
            "Predicted Cost"
        ] = np.round(

            cost_model[
                "full_predictions"
            ],

            2
        )

        if (

            task_types.get(
                "cost_target"
            )

            ==

            "regression"
        ):

            enterprise_df[
                "Cost Error"
            ] = (

                pd.to_numeric(

                    enterprise_df[
                        "Actual Cost"
                    ],

                    errors="coerce"
                )

                -

                pd.to_numeric(

                    enterprise_df[
                        "Predicted Cost"
                    ],

                    errors="coerce"
                )
            )

    except Exception as e:

        print(
            f"Cost Output Error -> {e}"
        )

# =========================================================
# SAFETY OUTPUTS
# =========================================================

if safety_model is not None:

    try:

        enterprise_df[
            "Actual Safety"
        ] = safety_model[
            "full_actual"
        ]

        enterprise_df[
            "Predicted Safety"
        ] = safety_model[
            "full_predictions"
        ]

        if (

            task_types.get(
                "safety_target"
            )

            ==

            "regression"
        ):

            enterprise_df[
                "Safety Error"
            ] = (

                pd.to_numeric(

                    enterprise_df[
                        "Actual Safety"
                    ],

                    errors="coerce"
                )

                -

                pd.to_numeric(

                    enterprise_df[
                        "Predicted Safety"
                    ],

                    errors="coerce"
                )
            )

    except Exception as e:

        print(
            f"Safety Output Error -> {e}"
        )

# =========================================================
# SUPPLIER INTELLIGENCE DIAGNOSTICS
# =========================================================

if not supplier_intelligence_df.empty:

    print("\n================================")
    print("SUPPLIER INTELLIGENCE")
    print("================================")
    print(
        f"Parts Analysed: "
        f"{len(supplier_intelligence_df)}"
    )
    print("================================\n")

# =========================================================
# DEBUG
# =========================================================

print("\n================================")
print("PROCUREMENT ENGINE")
print("================================")

if not supplier_intelligence_df.empty:

    print(
        "Parts Analysed:",
        len(
            supplier_intelligence_df
        )
    )

    print(
        "Average Score:",
        round(
            supplier_intelligence_df[
                "Procurement Score"
            ].mean(),
            2
        )
    )

    print(
        "Highest Score:",
        round(
            supplier_intelligence_df[
                "Procurement Score"
            ].max(),
            2
        )
    )

    print(
        "Lowest Score:",
        round(
            supplier_intelligence_df[
                "Procurement Score"
            ].min(),
            2
        )
    )

    print("\nRecommended Suppliers:")

    print(

        supplier_intelligence_df[
            "Recommended Supplier"
        ]

        .value_counts()
    )

else:

    print(
        "Supplier intelligence unavailable."
    )

print("================================\n")

# =========================================================
# ENTERPRISE OVERVIEW
# =========================================================

if selected_page == "Enterprise Overview":

    section_title(
        "Enterprise Dataset Intelligence"
    )

    # =====================================================
    # OVERVIEW METRICS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Rows",
            len(raw_business_df)
        )

    with col2:

        st.metric(
            "Columns",
            len(raw_business_df.columns)
        )

    with col3:

        st.metric(

            "Missing Values",

            validation_report
            .get(
                "missing_values",
                {}
            )
            .get(
                "total_missing",
                0
            )
        )

    with col4:

        st.metric(

            "Duplicate Rows",

            validation_report
            .get(
                "duplicates",
                {}
            )
            .get(
                "duplicate_rows",
                0
            )
        )

    section_divider()

    # =====================================================
    # ENTITY & SUPPLIER DETECTION
    # =====================================================

    st.subheader(
        "Enterprise Intelligence Configuration"
    )

    config_df = pd.DataFrame({

        "Configuration": [

            "Entity Column",

            "Supplier Column"
        ],

        "Detected Value": [

            entity_column,

            supplier_column
        ]
    })

    st.dataframe(

        config_df,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # DATASET SCHEMA
    # =====================================================

    st.subheader(
        "Dataset Schema"
    )

    schema_df = pd.DataFrame({

        "Column":
        raw_business_df.columns,

        "Data Type":
        raw_business_df.dtypes.astype(str),

        "Missing Values":
        raw_business_df.isnull().sum().values,

        "Unique Values":
        raw_business_df.nunique().values
    })

    st.dataframe(

        schema_df,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # TARGETS
    # =====================================================

    st.subheader(
        "AI Detected Targets"
    )

    target_df = pd.DataFrame({

        "Target Type": [

            "Demand Target",

            "Cost Target",

            "Safety Target"
        ],

        "Detected Column": [

            targets.get(
                "demand_target"
            ),

            targets.get(
                "cost_target"
            ),

            targets.get(
                "safety_target"
            )
        ],

        "Task Type": [

            task_types.get(
                "demand_target"
            ),

            task_types.get(
                "cost_target"
            ),

            task_types.get(
                "safety_target"
            )
        ]
    })

    st.dataframe(

        target_df,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # FEATURE INFORMATION
    # =====================================================

    st.subheader(
        "Feature Engineering Summary"
    )

    feature_df = pd.DataFrame({

        "Metric": [

            "Feature Count",

            "Target Count"
        ],

        "Value": [

            len(
                feature_columns
            ),

            len(
                target_columns
            )
        ]
    })

    st.dataframe(

        feature_df,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # PREPROCESSING DIAGNOSTICS
    # =====================================================

    st.subheader(
        "Preprocessing Diagnostics"
    )

    diagnostics_df = pd.DataFrame({

        "Metric":
        list(
            preprocessing_diagnostics.keys()
        ),

        "Value":
        list(
            preprocessing_diagnostics.values()
        )
    })

    st.dataframe(

        diagnostics_df,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # DATASET PREVIEW
    # =====================================================

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(

        raw_business_df.head(
            100
        ),

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # DOWNLOAD
    # =====================================================

    st.download_button(

        label=
        "Download Enterprise Dataset",

        data=
        raw_business_df.to_csv(
            index=False
        ),

        file_name=
        "enterprise_dataset.csv",

        mime=
        "text/csv"
    )

# =========================================================
# DEMAND INTELLIGENCE
# =========================================================

elif selected_page == "Demand Intelligence":

    section_title(
        "Demand Intelligence"
    )

    demand_target = targets.get(
        "demand_target"
    )

    demand_task = task_types.get(
        "demand_target"
    )

    st.info(

        f"""
Detected Demand Target:
{demand_target}

Detected Task Type:
{demand_task}
"""
    )

    # =====================================================
    # MODEL VALIDATION
    # =====================================================

    if demand_model is None:

        demand_error = (

            pipeline_output
            .get("errors", {})
            .get(
                "demand_model",
                "Unknown Error"
            )
        )

        st.error(

            f"""
Demand Model Unavailable

Reason:

{demand_error}
"""
        )

        st.stop()

    # =====================================================
    # MODEL INFO
    # =====================================================

    st.subheader(
        "Model Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(

            "Model",

            demand_model.get(
                "best_model_name",
                "Unknown"
            )
        )

    with col2:

        st.metric(

            "Task Type",

            demand_task
        )

    with col3:

        st.metric(

            "Records",

            len(
                demand_model[
                    "full_predictions"
                ]
            )
        )

    section_divider()

    metrics = demand_model.get(
        "metrics",
        {}
    )

    # =====================================================
    # KPI METRICS
    # =====================================================

    if demand_task == "regression":

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.metric(
                "R²",
                metrics.get(
                    "Test R2",
                    0
                )
            )

        with k2:
            st.metric(
                "MAE",
                metrics.get(
                    "MAE",
                    0
                )
            )

        with k3:
            st.metric(
                "RMSE",
                metrics.get(
                    "RMSE",
                    0
                )
            )

        with k4:
            st.metric(
                "MAPE (%)",
                metrics.get(
                    "MAPE",
                    0
                )
            )

    else:

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.metric(
                "Accuracy",
                metrics.get(
                    "Test Accuracy",
                    0
                )
            )

        with k2:
            st.metric(
                "Precision",
                metrics.get(
                    "Precision",
                    0
                )
            )

        with k3:
            st.metric(
                "Recall",
                metrics.get(
                    "Recall",
                    0
                )
            )

        with k4:
            st.metric(
                "F1 Score",
                metrics.get(
                    "F1 Score",
                    0
                )
            )

    section_divider()

    # =====================================================
    # RESULTS DATASET
    # =====================================================

    demand_results = (
        raw_business_df.copy()
    )

    demand_results[
        "Actual Demand"
    ] = demand_model[
        "full_actual"
    ]

    demand_results[
        "Predicted Demand"
    ] = demand_model[
        "full_predictions"
    ]

    # =====================================================
    # REGRESSION ANALYSIS
    # =====================================================

    if demand_task == "regression":

        demand_results[
            "Demand Error"
        ] = np.round(

            pd.to_numeric(

                demand_results[
                    "Predicted Demand"
                ],

                errors="coerce"
            )

            -

            pd.to_numeric(

                demand_results[
                    "Actual Demand"
                ],

                errors="coerce"
            ),

            2
        )

        demand_results[
            "Demand Accuracy (%)"
        ] = np.clip(

            100

            -

            (

                np.abs(

                    demand_results[
                        "Demand Error"
                    ]
                )

                /

                (

                    np.abs(

                        pd.to_numeric(

                            demand_results[
                                "Actual Demand"
                            ],

                            errors="coerce"
                        )
                    )

                    + 1e-6
                )

            ) * 100,

            0,

            100
        )

    else:

        demand_results[
            "Prediction Status"
        ] = np.where(

            demand_results[
                "Actual Demand"
            ].astype(str)

            ==

            demand_results[
                "Predicted Demand"
            ].astype(str),

            "Correct",

            "Mismatch"
        )

    # =====================================================
    # DEMAND SUPPLIER INTELLIGENCE
    # =====================================================

    if not supplier_intelligence_df.empty:

        st.subheader(
            "Best Demand Supplier By Part"
        )

        demand_supplier_df = (

            supplier_intelligence_df[
                [
                    "Part Name",
                    "Best Demand Supplier",
                    "Best Demand",
                    "Demand Score",
                    "Recommended Supplier",
                    "Recommended Demand",
                    "Procurement Score"
                ]
            ]

            .copy()

            .sort_values(
                by="Part Name"
            )

            .reset_index(
                drop=True
            )
        )

        st.dataframe(

            demand_supplier_df,

            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(

                "Parts Analysed",

                len(
                    demand_supplier_df
                )
            )

        with col2:

            st.metric(

                "Unique Demand Leaders",

                demand_supplier_df[
                    "Best Demand Supplier"
                ].nunique()
            )

        section_divider()

    else:

        st.info(

            "Supplier intelligence data unavailable."

        )

        section_divider()

    supplier_count = (
            demand_supplier_df["Best Demand Supplier"]
            .value_counts()
            .reset_index())
        
    supplier_count.columns = ["Supplier","Count"]
        
    fig = px.bar(supplier_count,
                     x="Supplier",
                     y="Count",
                     title="Demand Leaders By Supplier")
        
    st.plotly_chart(
            fig,
            use_container_width=True)

    # =====================================================
    # RESULTS TABLE
    # =====================================================

    st.subheader(
        "Demand Intelligence Results"
    )

    st.dataframe(

        demand_results.head(100),

        use_container_width=True
    )

    section_divider()

    st.subheader(
    "Prediction Analysis")
    
    if demand_task == "regression":
        fig = px.scatter(
            demand_results,
            x="Actual Demand",
            y="Predicted Demand",
            title="Actual vs Predicted Demand")
        
        st.plotly_chart(
            fig,
            use_container_width=True)
        
        fig = px.histogram(
            demand_results,
            x="Demand Error",
            nbins=30,
            title="Demand Error Distribution")
        
        st.plotly_chart(
            fig,
            use_container_width=True)
    
    else:
        prediction_summary = (
            demand_results["Prediction Status"]
            .value_counts()
            .reset_index())
        
        prediction_summary.columns = [
            "Status","Count"]
        
        fig = px.pie(
            prediction_summary,
            names="Status",
            values="Count",
            title="Demand Classification Performance"
        )
        st.plotly_chart(
            fig,use_container_width=True)

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    feature_importance = demand_model.get(
    "feature_importance")
    
    st.write(
    "Best Model:",
    demand_model.get(
        "best_model_name"))
    
    st.dataframe(
    feature_importance)
    
    if (feature_importance is not None and len(feature_importance) > 0):
        
        st.subheader("Feature Importance")
        
        st.plotly_chart(
            feature_importance_chart(feature_importance),use_container_width=True)

    section_divider()

    # =====================================================
    # DOWNLOAD
    # =====================================================

    st.download_button(

        label=
        "Download Demand Intelligence CSV",

        data=
        demand_results.to_csv(
            index=False
        ),

        file_name=
        "demand_intelligence_results.csv",

        mime=
        "text/csv"
    )

# =========================================================
# COST INTELLIGENCE
# =========================================================

elif selected_page == "Cost Intelligence":

    section_title(
        "Cost Intelligence"
    )

    cost_target = targets.get(
        "cost_target"
    )

    cost_task = task_types.get(
        "cost_target"
    )

    st.info(

        f"""
Detected Cost Target:
{cost_target}

Detected Task Type:
{cost_task}
"""
    )

    # =====================================================
    # MODEL VALIDATION
    # =====================================================

    if cost_model is None:

        cost_error = (

            pipeline_output
            .get("errors", {})
            .get(
                "cost_model",
                "Unknown Error"
            )
        )

        st.error(

            f"""
Cost Model Unavailable

Reason:

{cost_error}
"""
        )

        st.stop()

    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    st.subheader(
        "Model Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(

            "Model",

            cost_model.get(
                "best_model_name",
                "Unknown"
            )
        )

    with col2:

        st.metric(

            "Task Type",

            cost_task
        )

    with col3:

        st.metric(

            "Records",

            len(
                cost_model[
                    "full_predictions"
                ]
            )
        )

    section_divider()

    metrics = cost_model.get(
        "metrics",
        {}
    )

    # =====================================================
    # KPI METRICS
    # =====================================================

    if cost_task == "regression":

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.metric(
                "R²",
                metrics.get(
                    "Test R2",
                    0
                )
            )

        with k2:
            st.metric(
                "MAE",
                metrics.get(
                    "MAE",
                    0
                )
            )

        with k3:
            st.metric(
                "RMSE",
                metrics.get(
                    "RMSE",
                    0
                )
            )

        with k4:
            st.metric(
                "MAPE (%)",
                metrics.get(
                    "MAPE",
                    0
                )
            )

    else:

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.metric(
                "Accuracy",
                metrics.get(
                    "Test Accuracy",
                    0
                )
            )

        with k2:
            st.metric(
                "Precision",
                metrics.get(
                    "Precision",
                    0
                )
            )

        with k3:
            st.metric(
                "Recall",
                metrics.get(
                    "Recall",
                    0
                )
            )

        with k4:
            st.metric(
                "F1 Score",
                metrics.get(
                    "F1 Score",
                    0
                )
            )

    section_divider()

    # =====================================================
    # RESULTS DATASET
    # =====================================================

    cost_results = (
        raw_business_df.copy()
    )

    cost_results[
        "Actual Cost"
    ] = cost_model[
        "full_actual"
    ]

    cost_results[
        "Predicted Cost"
    ] = cost_model[
        "full_predictions"
    ]

    # =====================================================
    # REGRESSION ANALYSIS
    # =====================================================

    if cost_task == "regression":

        cost_results[
            "Cost Error"
        ] = np.round(

            pd.to_numeric(

                cost_results[
                    "Predicted Cost"
                ],

                errors="coerce"
            )

            -

            pd.to_numeric(

                cost_results[
                    "Actual Cost"
                ],

                errors="coerce"
            ),

            2
        )

        cost_results[
            "Cost Accuracy (%)"
        ] = np.clip(

            100

            -

            (

                np.abs(

                    cost_results[
                        "Cost Error"
                    ]
                )

                /

                (

                    np.abs(

                        pd.to_numeric(

                            cost_results[
                                "Actual Cost"
                            ],

                            errors="coerce"
                        )
                    )

                    + 1e-6
                )

            ) * 100,

            0,

            100
        )

    else:

        cost_results[
            "Prediction Status"
        ] = np.where(

            cost_results[
                "Actual Cost"
            ].astype(str)

            ==

            cost_results[
                "Predicted Cost"
            ].astype(str),

            "Correct",

            "Mismatch"
        )

    # =====================================================
    # COST SUPPLIER INTELLIGENCE
    # =====================================================

    if not supplier_intelligence_df.empty:

        st.subheader(
            "Best Cost Supplier By Part"
        )

        best_cost_df = (

            supplier_intelligence_df[
                [
                    "Part Name",
                    "Best Cost Supplier",
                    "Best Cost",
                    "Cost Score",
                    "Recommended Supplier",
                    "Recommended Cost",
                    "Procurement Score"
                ]
            ]

            .copy()

            .sort_values(
                by="Part Name"
            )

            .reset_index(
                drop=True
            )
        )

        st.dataframe(

            best_cost_df,

            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(

                "Parts Analysed",

                len(
                    best_cost_df
                )
            )

        with col2:

            st.metric(

                "Unique Cost Leaders",

                best_cost_df[
                    "Best Cost Supplier"
                ].nunique()
            )

        section_divider()

    else:

        st.info(

            "Supplier intelligence data unavailable."

        )

        section_divider()

    st.subheader("Prediction Analysis")
    
    if cost_task == "regression":
        fig = px.scatter(
            cost_results,
            x="Actual Cost",
            y="Predicted Cost",
            title="Actual vs Predicted Cost"
        )
        st.plotly_chart(
            fig,
            use_container_width=True)
        fig = px.histogram(
            cost_results,
            x="Cost Error",
            nbins=30,
            title="Cost Error Distribution")
        st.plotly_chart(
            fig,
            use_container_width=True
        )
    else:
        prediction_summary = (
            cost_results["Prediction Status"]
            .value_counts()
            .reset_index())
        prediction_summary.columns = ["Status","Count"]
        fig = px.pie(
            prediction_summary,
            names="Status",
            values="Count",
            title="Cost Classification Performance")
        st.plotly_chart(
            fig,
            use_container_width=True)
        
    supplier_count = (
            best_cost_df["Best Cost Supplier"]
            .value_counts()
            .reset_index())
    supplier_count.columns = ["Supplier","Count"]
    fig = px.bar(
            supplier_count,
            x="Supplier",
            y="Count",
            title="Cost Leaders By Supplier")
    st.plotly_chart(
            fig,
            use_container_width=True)

    # =====================================================
    # RESULTS TABLE
    # =====================================================

    st.subheader(
        "Cost Intelligence Results"
    )

    st.dataframe(

        cost_results.head(100),

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    feature_importance = cost_model.get(
        "feature_importance"
    )

    if (

        feature_importance is not None

        and

        len(feature_importance) > 0
    ):

        st.subheader(
            "Feature Importance"
        )

        st.plotly_chart(

            feature_importance_chart(

                feature_importance
            ),

            use_container_width=True
        )

    section_divider()

    # =====================================================
    # DOWNLOAD
    # =====================================================

    st.download_button(

        label=
        "Download Cost Intelligence CSV",

        data=
        cost_results.to_csv(
            index=False
        ),

        file_name=
        "cost_intelligence_results.csv",

        mime=
        "text/csv"
    )

# =========================================================
# SAFETY INTELLIGENCE
# =========================================================

elif selected_page == "Safety Intelligence":

    section_title(
        "Safety Intelligence"
    )

    safety_target = targets.get(
        "safety_target"
    )

    safety_task = task_types.get(
        "safety_target"
    )

    # =====================================================
    # TARGET INFO
    # =====================================================

    st.info(

        f"""
Detected Safety Target:
{safety_target}

Detected Task Type:
{safety_task}
"""
    )

    # =====================================================
    # MODEL FAILURE
    # =====================================================

    if safety_model is None:

        safety_error = (

            pipeline_output

            .get("errors", {})

            .get(

                "safety_model",

                "Unknown Error"
            )
        )

        st.error(

            f"""
Safety model unavailable.

Reason:

{safety_error}
"""
        )

        st.stop()

    # =====================================================
    # MODEL INFO
    # =====================================================

    st.subheader(
        "Model Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(

            "Model",

            safety_model.get(

                "best_model_name",

                "Unknown"
            )
        )

    with col2:

        st.metric(

            "Task Type",

            safety_task
        )

    with col3:

        st.metric(

            "Records",

            len(

                safety_model.get(

                    "full_actual",

                    []
                )
            )
        )

    metrics = safety_model.get(
        "metrics",
        {}
    )

    section_divider()

    # =====================================================
    # KPI CARDS
    # =====================================================

    if safety_task == "regression":

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.metric(
                "R²",
                metrics.get(
                    "Test R2",
                    0
                )
            )

        with k2:
            st.metric(
                "MAE",
                metrics.get(
                    "MAE",
                    0
                )
            )

        with k3:
            st.metric(
                "RMSE",
                metrics.get(
                    "RMSE",
                    0
                )
            )

        with k4:
            st.metric(
                "MAPE (%)",
                metrics.get(
                    "MAPE",
                    0
                )
            )

    else:

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.metric(
                "Accuracy",
                metrics.get(
                    "Test Accuracy",
                    0
                )
            )

        with k2:
            st.metric(
                "Precision",
                metrics.get(
                    "Precision",
                    0
                )
            )

        with k3:
            st.metric(
                "Recall",
                metrics.get(
                    "Recall",
                    0
                )
            )

        with k4:
            st.metric(
                "F1 Score",
                metrics.get(
                    "F1 Score",
                    0
                )
            )

    section_divider()

    st.subheader(
    "Prediction Analysis")
    

    # =====================================================
    # BUILD RESULTS
    # =====================================================

    safe_length = max(

        1,

        min(

            len(raw_business_df),

            len(

                safety_model.get(

                    "full_actual",

                    []
                )
            ),

            len(

                safety_model.get(

                    "full_predictions",

                    []
                )
            )
        )
    )

    safety_results = (

        raw_business_df

        .iloc[:safe_length]

        .copy()
    )

    safety_results[
        "Actual Safety"
    ] = np.array(

        safety_model.get(

            "full_actual",

            []
        )

    )[:safe_length]

    safety_results[
        "Predicted Safety"
    ] = np.array(

        safety_model.get(

            "full_predictions",

            []
        )

    )[:safe_length]

    # =====================================================
    # REGRESSION ANALYSIS
    # =====================================================

    if safety_task == "regression":

        safety_results[
            "Safety Error"
        ] = np.round(

            pd.to_numeric(

                safety_results[
                    "Predicted Safety"
                ],

                errors="coerce"
            )

            -

            pd.to_numeric(

                safety_results[
                    "Actual Safety"
                ],

                errors="coerce"
            ),

            2
        )

        safety_results[
            "Safety Accuracy (%)"
        ] = np.clip(

            100

            -

            (

                np.abs(

                    safety_results[
                        "Safety Error"
                    ]
                )

                /

                (

                    np.abs(

                        pd.to_numeric(

                            safety_results[
                                "Actual Safety"
                            ],

                            errors="coerce"
                        )

                    )

                    +

                    1e-6
                )

            ) * 100,

            0,

            100
        )

    # =====================================================
    # CLASSIFICATION ANALYSIS
    # =====================================================

    else:

        safety_results[
            "Prediction Status"
        ] = np.where(

            safety_results[
                "Actual Safety"
            ].astype(str)

            ==

            safety_results[
                "Predicted Safety"
            ].astype(str),

            "Correct",

            "Mismatch"
        )

    # =====================================================
    # SAFETY SUPPLIER INTELLIGENCE
    # =====================================================

    if not supplier_intelligence_df.empty:

        st.subheader(
            "Best Safety Supplier By Part"
        )

        safety_supplier_df = (

            supplier_intelligence_df[
                [
                    "Part Name",
                    "Best Safety Supplier",
                    "Safety Score",
                    "Recommended Supplier",
                    "Recommended Safety",
                    "Procurement Score"
                ]
            ]

            .copy()

            .sort_values(
                by="Part Name"
            )

            .reset_index(
                drop=True
            )
        )

        st.dataframe(

            safety_supplier_df,

            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(

                "Parts Analysed",

                len(
                    safety_supplier_df
                )
            )

        with col2:

            st.metric(

                "Unique Safety Leaders",

                safety_supplier_df[
                    "Best Safety Supplier"
                ].nunique()
            )

        section_divider()

    else:

        st.info(

            "Supplier intelligence data unavailable."

        )

        section_divider()

    # =====================================================
    # SAFETY RESULTS
    # =====================================================

    st.subheader(
        "Safety Intelligence Results"
    )

    st.dataframe(

        safety_results.head(100),

        use_container_width=True
    )

    section_divider()

    if safety_task == "regression":
        fig = px.scatter(
            safety_results,
            x="Actual Safety",
            y="Predicted Safety",
            title="Actual vs Predicted Safety")
        st.plotly_chart(
            fig,
            use_container_width=True)
        fig = px.histogram(
            safety_results,
            x="Safety Error",
            nbins=30,
            title="Safety Error Distribution")
        st.plotly_chart(
            fig,use_container_width=True)
    
    else:
        prediction_summary = (
            safety_results["Prediction Status"]
            .value_counts()
            .reset_index())
        prediction_summary.columns = ["Status","Count"]
        fig = px.pie(
            prediction_summary,
            names="Status",
            values="Count",
            title="Safety Classification Performance")
        st.plotly_chart(
            fig,
            use_container_width=True)
        
        class_distribution = (
            safety_results["Predicted Safety"].value_counts()
            .reset_index())
        
        class_distribution.columns = [
            "Class","Count"]
        
        fig = px.bar(
            class_distribution,
            x="Class",
            y="Count",
            title="Predicted Safety Distribution")
        st.plotly_chart(
            fig,use_container_width=True)

    # =====================================================
    # SUMMARY KPIs
    # =====================================================

    if safety_task == "regression":

        k1, k2, k3 = st.columns(3)

        with k1:

            st.metric(

                "Average Actual",

                round(

                    pd.to_numeric(

                        safety_results[
                            "Actual Safety"
                        ],

                        errors="coerce"
                    ).mean(),

                    2
                )
            )

        with k2:

            st.metric(

                "Average Prediction",

                round(

                    pd.to_numeric(

                        safety_results[
                            "Predicted Safety"
                        ],

                        errors="coerce"
                    ).mean(),

                    2
                )
            )

        with k3:

            st.metric(

                "Average Error",

                round(

                    np.abs(

                        safety_results[
                            "Safety Error"
                        ]

                    ).mean(),

                    2
                )
            )

    else:

        total_records = len(
            safety_results
        )

        correct_predictions = len(

            safety_results[

                safety_results[
                    "Prediction Status"
                ]

                ==

                "Correct"
            ]
        )

        prediction_accuracy = round(

            (

                correct_predictions

                /

                max(
                    total_records,
                    1
                )

            ) * 100,

            2
        )

        k1, k2, k3 = st.columns(3)

        with k1:
            st.metric(
                "Records",
                total_records
            )

        with k2:
            st.metric(
                "Correct",
                correct_predictions
            )

        with k3:
            st.metric(
                "Prediction Accuracy",
                f"{prediction_accuracy}%"
            )

    section_divider()

    

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    feature_importance = safety_model.get(
        "feature_importance"
    )

    if (

        feature_importance is not None

        and

        len(feature_importance) > 0

    ):

        st.subheader(
            "Feature Importance"
        )

        st.plotly_chart(

            feature_importance_chart(

                feature_importance
            ),

            use_container_width=True
        )

    section_divider()

    # =====================================================
    # DOWNLOAD
    # =====================================================

    st.download_button(

        label="Download Safety Intelligence CSV",

        data=safety_results.copy().to_csv(

            index=False
        ),

        file_name=

        "safety_intelligence_results.csv",

        mime="text/csv"
    )

# =========================================================
# PROCUREMENT INTELLIGENCE
# =========================================================

elif selected_page == "Procurement Intelligence":

    section_title(
        "Procurement Intelligence"
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    if supplier_intelligence_df.empty:
        
        
        st.error(
            """
            Supplier intelligence has not been generated.
            Please verify model training.
            
            """
        )
        
        st.stop()

    # =====================================================
    # SCORE SERIES
    # =====================================================

    score_series = pd.to_numeric(

    supplier_intelligence_df[
        "Procurement Score"
    ],

    errors="coerce"
    
    ).fillna(0)

    # =====================================================
    # KPI CARDS
    # =====================================================

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:

        st.metric(

            "Average Score",

            round(
                score_series.mean(),
                2
            )
        )

    with k2:

        st.metric(

            "Best Score",

            round(
                score_series.max(),
                2
            )
        )

    with k3:

        st.metric(

            "Lowest Score",

            round(
                score_series.min(),
                2
            )
        )

    with k4:

        recommended_suppliers = (
            
            supplier_intelligence_df[
        "Recommended Supplier"
            ].nunique())
        
        st.metric(
            "Recommended Suppliers",
            
            recommended_suppliers)

    with k5:

        st.metric(
            "Procurement Situations",
            supplier_intelligence_df[
            "Procurement Situation"].nunique())

    with k6:

        st.metric(

        "In-House Opportunities",

        (supplier_intelligence_df["Procurement Mode"] == "In-House").sum())

    section_divider()

    # =====================================================
    # CHARTS
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        try:

            score_chart = (

                procurement_score_chart(

                    score_series
                )
            )

            if score_chart is not None:

                st.plotly_chart(

                    score_chart,

                    use_container_width=True
                )

        except Exception as e:

            st.warning(
                f"Score Chart Error: {e}"
            )

    with col2:

        try:

            tier_chart = (
                
                supplier_ranking_chart(
                    
                    supplier_intelligence_df)
            )

            if tier_chart is not None:

                st.plotly_chart(

                    tier_chart,

                    use_container_width=True
                )

        except Exception as e:

            st.warning(
                f"Tier Chart Error: {e}"
            )

    section_divider()

    # =====================================================
    # TOP PROCUREMENT OPPORTUNITIES
    # =====================================================

    st.subheader(
    "Best Supplier Recommendations By Part")
        
    top_procurement = (
        
        supplier_intelligence_df
        
        .sort_values(

        "Part Name"))
        
    st.dataframe(
        
        top_procurement[
        [
            "Part Name",
            "Best Demand Supplier",
            "Best Demand",
            "Best Cost Supplier",
            "Best Cost",
            "Best Safety Supplier",
            "Safety Score",
            "Procurement Situation",
            "Procurement Mode",
            "Recommended Supplier",
            "Recommended Demand",
            "Recommended Cost",
            "Recommended Safety",
            "Procurement Score"
        ]
    ],
        use_container_width=True)
        
    section_divider()

    # =====================================================
    # SUPPLIER RANKING
    # =====================================================

    st.subheader(
    "Recommended Supplier Ranking")
        
    supplier_summary = (
        
        supplier_intelligence_df
        
        .groupby(
        "Recommended Supplier")
        
        .agg({

        "Procurement Score":
        "mean",
        "Part Name":"count"})
        
        .reset_index()
        
        .sort_values(

        "Procurement Score",
        ascending=False

        ))

    supplier_summary.columns = [

    "Recommended Supplier",

    "Average Procurement Score",

    "Parts Won"]
        
    supplier_summary["Average Procurement Score"] = supplier_summary["Average Procurement Score"].round(2)
        
    st.dataframe(
        supplier_summary,
        use_container_width=True)
    
    section_divider()

    # =====================================================
    # CONSOLIDATED INTELLIGENCE
    # =====================================================

    st.subheader(
    "Consolidated Procurement Intelligence")
        
    st.dataframe(
        supplier_intelligence_df,
        use_container_width=True)
        
    section_divider()

    # =====================================================
    # SCORE DISTRIBUTION
    # =====================================================

    st.subheader(
        "Procurement Score Distribution"
    )

    score_distribution = (

        score_series

        .describe()

        .reset_index()
    )

    score_distribution.columns = [

        "Metric",

        "Value"
    ]

    st.dataframe(

        score_distribution,

        use_container_width=True
    )

    section_divider()

    st.subheader(
    "Procurement Situation Distribution")
    
    situation_summary = (supplier_intelligence_df["Procurement Situation"]
                         .value_counts()
                         .reset_index())
    
    situation_summary.columns = ["Situation","Count"]
    
    st.dataframe(
        situation_summary,
        use_container_width=True)

    st.subheader(
    "Procurement Mode Distribution")
    
    mode_summary = (supplier_intelligence_df["Procurement Mode"]
                    .value_counts().reset_index())
    
    mode_summary.columns = ["Procurement Mode","Count"]
    
    st.dataframe(mode_summary,use_container_width=True)

    fig = px.histogram(
        supplier_intelligence_df,
        x="Procurement Score",
        nbins=15,
        title="Procurement Score Histogram")
    
    st.plotly_chart(
        fig,
        use_container_width=True)

    fig = px.pie(
        situation_summary,
        names="Situation",
        values="Count",
        title="Procurement Situation Distribution")
    
    st.plotly_chart(
        fig,
        use_container_width=True)

    fig = px.bar(
        mode_summary,
        x="Procurement Mode",
        y="Count",
        title="Procurement Mode Distribution")
    
    st.plotly_chart(
        fig,
        use_container_width=True)

    # =====================================================
    # DOWNLOAD
    # =====================================================

    st.download_button(

        label=
        "Download Procurement Intelligence CSV",

        data=
        supplier_intelligence_df.to_csv(
            index=False
        ),

        file_name=
        "procurement_intelligence.csv",

        mime=
        "text/csv"
    )

# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif selected_page == "Model Performance":

    section_title(
        "Enterprise Model Intelligence"
    )

    # =====================================================
    # MODEL MAPPING
    # =====================================================

    model_mapping = {

        "Demand":
        demand_model,

        "Cost":
        cost_model,

        "Safety":
        safety_model
    }

    performance_rows = []

    diagnostics_rows = []

    # =====================================================
    # BUILD PERFORMANCE DATA
    # =====================================================

    for pipeline_name, model_output in model_mapping.items():

        if model_output is None:

            continue

        metrics = model_output.get(
            "metrics",
            {}
        )

        row = {

            "Pipeline":
            pipeline_name,

            "Selected Model":
            model_output.get(
                "best_model_name",
                "N/A"
            )
        }

        row.update(metrics)

        performance_rows.append(
            row
        )

        diagnostics = model_output.get(
            "diagnostics",
            {}
        )

        diagnostics_rows.append({

            "Pipeline":
            pipeline_name,

            "Rows":
            diagnostics.get(
                "rows",
                len(
                    model_output.get(
                        "full_predictions",
                        []
                    )
                )
            ),

            "Features":
            diagnostics.get(
                "features",
                len(
                    feature_columns
                )
            ),

            "Best Score":
            diagnostics.get(
                "best_score",
                "N/A"
            )
        })

    # =====================================================
    # VALIDATION
    # =====================================================

    if len(performance_rows) == 0:

        st.warning(
            "No trained models available."
        )

        st.stop()

    performance_df = pd.DataFrame(
        performance_rows
    )

    diagnostics_df = pd.DataFrame(
        diagnostics_rows
    )

    # =====================================================
    # MODEL SUMMARY
    # =====================================================

    st.subheader(
        "Selected Enterprise Models"
    )

    k1, k2, k3 = st.columns(3)

    with k1:

        st.metric(

            "Demand Model",

            demand_model.get(
                "best_model_name",
                "N/A"
            ) if demand_model else "N/A"
        )

    with k2:

        st.metric(

            "Cost Model",

            cost_model.get(
                "best_model_name",
                "N/A"
            ) if cost_model else "N/A"
        )

    with k3:

        st.metric(

            "Safety Model",

            safety_model.get(
                "best_model_name",
                "N/A"
            ) if safety_model else "N/A"
        )

    section_divider()

    # =====================================================
    # ENTERPRISE CONFIGURATION
    # =====================================================

    st.subheader(
        "Enterprise Configuration"
    )

    config_df = pd.DataFrame({

        "Configuration": [

            "Entity Column",

            "Supplier Column",

            "Feature Count",

            "Target Count"
        ],

        "Value": [

            entity_column,

            supplier_column,

            len(
                feature_columns
            ),

            len(
                target_columns
            )
        ]
    })

    st.dataframe(

        config_df,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # MODEL DIAGNOSTICS
    # =====================================================

    st.subheader(
        "Model Diagnostics"
    )

    st.dataframe(

        diagnostics_df,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # PERFORMANCE METRICS
    # =====================================================

    st.subheader(
        "Performance Metrics"
    )

    st.dataframe(

        performance_df,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.subheader(
        "Feature Importance Analysis"
    )

    for pipeline_name, model_output in model_mapping.items():

        if model_output is None:

            continue

        feature_importance = (

            model_output.get(
                "feature_importance"
            )
        )

        if (

            feature_importance is not None

            and

            len(feature_importance) > 0

        ):

            st.markdown(

                f"### {pipeline_name} Model"
            )

            try:

                st.plotly_chart(

                    feature_importance_chart(

                        feature_importance
                    ),

                    use_container_width=True
                )

            except Exception as e:

                st.warning(

                    f"{pipeline_name} Feature Importance Error: {e}"
                )

    section_divider()

    # =====================================================
    # TRAINING DATA SUMMARY
    # =====================================================

    st.subheader(
        "Training Dataset Summary"
    )

    training_summary = pd.DataFrame({

        "Metric": [

            "Rows",

            "Columns",

            "Features",

            "Targets"
        ],

        "Value": [

            len(
                processed_df
            ),

            len(
                processed_df.columns
            ),

            len(
                feature_columns
            ),

            len(
                target_columns
            )
        ]
    })

    st.dataframe(

        training_summary,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # PREPROCESSING DIAGNOSTICS
    # =====================================================

    st.subheader(
        "Preprocessing Diagnostics"
    )

    preprocessing_df = pd.DataFrame({

        "Metric":
        list(
            preprocessing_diagnostics.keys()
        ),

        "Value":
        list(
            preprocessing_diagnostics.values()
        )
    })

    st.dataframe(

        preprocessing_df,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # PIPELINE ERRORS
    # =====================================================

    st.subheader(
        "Pipeline Errors"
    )

    pipeline_errors = (

        pipeline_output.get(
            "errors",
            {}
        )
    )

    if pipeline_errors:

        st.json(
            pipeline_errors
        )

    else:

        st.success(
            "No pipeline errors detected."
        )

    section_divider()

    # =====================================================
    # PIPELINE DIAGNOSTICS
    # =====================================================

    st.subheader(
        "Pipeline Diagnostics"
    )

    st.json(
        pipeline_diagnostics
    )

    section_divider()

    # =====================================================
    # SUPPLIER INTELLIGENCE SUMMARY
    # =====================================================

    if not supplier_intelligence_df.empty:
        
        st.subheader(
        "Supplier Intelligence Summary"
    )
        
        st.dataframe(

        supplier_intelligence_df.head(20),

        use_container_width=True
    )
        
        section_divider()

    # =====================================================
    # DOWNLOAD
    # =====================================================

    st.download_button(

        label=
        "Download Model Performance CSV",

        data=
        performance_df.to_csv(
            index=False
        ),

        file_name=
        "model_performance.csv",

        mime=
        "text/csv"
    )

# =========================================================
# DATASET INTELLIGENCE
# =========================================================

elif selected_page == "Dataset Intelligence":

    section_title(
        "Dataset Intelligence"
    )

    # =====================================================
    # DATASET OVERVIEW
    # =====================================================

    row_count = len(
        raw_business_df
    )

    column_count = len(
        raw_business_df.columns
    )

    numeric_count = len(

        raw_business_df.select_dtypes(

            include=np.number

        ).columns
    )

    categorical_count = len(

        raw_business_df.select_dtypes(

            exclude=np.number

        ).columns
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Rows",
            row_count
        )

    with col2:

        st.metric(
            "Columns",
            column_count
        )

    with col3:

        st.metric(
            "Numeric Columns",
            numeric_count
        )

    with col4:

        st.metric(
            "Categorical Columns",
            categorical_count
        )

    section_divider()

    # =====================================================
    # ENTITY & SUPPLIER DETECTION
    # =====================================================

    st.subheader(
        "Enterprise Intelligence Configuration"
    )

    config_df = pd.DataFrame({

        "Configuration": [

            "Entity Column",

            "Supplier Column"
        ],

        "Value": [

            entity_column,

            supplier_column
        ]
    })

    st.dataframe(

        config_df,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # DATASET PREVIEW
    # =====================================================

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(

        raw_business_df.head(100),

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # DATASET SCHEMA
    # =====================================================

    st.subheader(
        "Dataset Schema"
    )

    schema_df = pd.DataFrame({

        "Column":
        raw_business_df.columns,

        "Data Type":
        raw_business_df.dtypes.astype(str),

        "Missing Values":
        raw_business_df.isnull().sum().values,

        "Unique Values":
        raw_business_df.nunique().values
    })

    st.dataframe(

        schema_df,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # TARGET DETECTION
    # =====================================================

    st.subheader(
        "Detected Enterprise Targets"
    )

    targets_df = pd.DataFrame({

        "Target Type": [

            "Demand Target",

            "Cost Target",

            "Safety Target"
        ],

        "Detected Column": [

            targets.get(
                "demand_target"
            ),

            targets.get(
                "cost_target"
            ),

            targets.get(
                "safety_target"
            )
        ],

        "Task Type": [

            task_types.get(
                "demand_target"
            ),

            task_types.get(
                "cost_target"
            ),

            task_types.get(
                "safety_target"
            )
        ]
    })

    st.dataframe(

        targets_df,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # SUPPLIER INTELLIGENCE SUMMARY
    # =====================================================

    if not supplier_intelligence_df.empty:
        
        st.subheader(
        "Supplier Intelligence Summary")
        
        summary_df = pd.DataFrame({

        "Metric": [

            "Parts Analysed",

            "Best Demand Suppliers",

            "Best Cost Suppliers",

            "Best Safety Suppliers",

            "Recommended Suppliers"
        ],

        "Value": [

            len(
                supplier_intelligence_df
            ),

            supplier_intelligence_df[
                "Best Demand Supplier"
            ].nunique(),

            supplier_intelligence_df[
                "Best Cost Supplier"
            ].nunique(),

            supplier_intelligence_df[
                "Best Safety Supplier"
            ].nunique(),

            supplier_intelligence_df[
                "Recommended Supplier"
            ].nunique()]})
        
        st.dataframe(
        summary_df,
        use_container_width=True)
        
        section_divider()

    # =====================================================
    # FEATURE SUMMARY
    # =====================================================

    st.subheader(
        "Feature Summary"
    )

    feature_summary = pd.DataFrame({

        "Metric": [

            "Feature Count",

            "Target Count"
        ],

        "Value": [

            len(
                feature_columns
            ),

            len(
                target_columns
            )
        ]
    })

    st.dataframe(

        feature_summary,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # FEATURE LIST
    # =====================================================

    st.subheader(
        "Feature Columns"
    )

    feature_df = pd.DataFrame({

        "Feature Columns":
        feature_columns
    })

    st.dataframe(

        feature_df,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # DATA QUALITY REPORT
    # =====================================================

    st.subheader(
        "Data Quality Report"
    )

    missing_values = (

        validation_report

        .get(
            "missing_values",
            {}
        )

        .get(
            "total_missing",
            0
        )
    )

    duplicate_rows = (

        validation_report

        .get(
            "duplicates",
            {}
        )

        .get(
            "duplicate_rows",
            0
        )
    )

    quality_df = pd.DataFrame({

        "Metric": [

            "Missing Values",

            "Duplicate Rows"
        ],

        "Value": [

            missing_values,

            duplicate_rows
        ]
    })

    st.dataframe(

        quality_df,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # PREPROCESSING DIAGNOSTICS
    # =====================================================

    st.subheader(
        "Preprocessing Diagnostics"
    )

    preprocessing_df = pd.DataFrame({

        "Metric":
        list(
            preprocessing_diagnostics.keys()
        ),

        "Value":
        list(
            preprocessing_diagnostics.values()
        )
    })

    st.dataframe(

        preprocessing_df,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # PIPELINE DIAGNOSTICS
    # =====================================================

    st.subheader(
        "Pipeline Diagnostics"
    )

    if pipeline_diagnostics:

        st.json(
            pipeline_diagnostics
        )

    else:

        st.info(
            "No diagnostics available."
        )

    section_divider()

    # =====================================================
    # PIPELINE ERRORS
    # =====================================================

    st.subheader(
        "Pipeline Errors"
    )

    if pipeline_errors:

        st.json(
            pipeline_errors
        )

    else:

        st.success(
            "No pipeline errors detected."
        )

    section_divider()

    # =====================================================
    # TRAINING SUMMARY
    # =====================================================

    st.subheader(
        "Training Summary"
    )

    training_df = pd.DataFrame({

        "Metric": [

            "Processed Rows",

            "Processed Columns",

            "Features",

            "Targets"
        ],

        "Value": [

            len(
                processed_df
            ),

            len(
                processed_df.columns
            ),

            len(
                feature_columns
            ),

            len(
                target_columns
            )
        ]
    })

    st.dataframe(

        training_df,

        use_container_width=True
    )

    section_divider()

    # =====================================================
    # DOWNLOAD
    # =====================================================

    st.download_button(

        label=
        "Download Dataset CSV",

        data=
        raw_business_df.to_csv(
            index=False
        ),

        file_name=
        "enterprise_dataset.csv",

        mime=
        "text/csv"
    )