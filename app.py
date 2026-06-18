import os
import pandas as pd

# ==========================================
# PREPROCESSING
# ==========================================

from preprocessing.detect_schema import (
    detect_schema
)

from preprocessing.clean_data import (
    clean_data
)

from preprocessing.outlier_handler import (
    remove_outliers
)

from preprocessing.feature_builder import (
    build_features
)

from preprocessing.encode_features import (
    encode_features
)

from preprocessing.scale_features import (
    scale_features
)

from preprocessing.time_series_features import (
    build_time_features
)

# ==========================================
# MODELS
# ==========================================

from models.train_model import (
    train_model
)

from models.forecasting_models import (
    train_forecasting_model
)

from models.decision_engine import (
    generate_decision
)

# ==========================================
# UTILS
# ==========================================

from utils.validation import (
    validate_dataset
)

from utils.helpers import (

    clean_column_names,

    remove_constant_columns,

    optimize_memory
)

# ==========================================
# MAIN APPLICATION
# ==========================================

def main():

    print(
        "\n====================================="
    )

    print(
        "INDUSTRIAL AI DECISION PLATFORM"
    )

    print(
        "=====================================\n"
    )

    # ======================================
    # DATASET PATH
    # ======================================

    dataset_path = input(
        "Enter dataset path: "
    )

    # ======================================
    # CHECK FILE EXISTS
    # ======================================

    if not os.path.exists(
        dataset_path
    ):

        print(
            "\nDataset file not found!"
        )

        return

    # ======================================
    # LOAD DATASET
    # ======================================

    try:

        if dataset_path.endswith(".csv"):

            df = pd.read_csv(
                dataset_path
            )

        elif dataset_path.endswith(".xlsx"):

            df = pd.read_excel(
                dataset_path
            )

        else:

            print(
                "\nUnsupported file format!"
            )

            return

    except Exception as e:

        print(
            f"\nDataset loading failed: {e}"
        )

        return

    # ======================================
    # DATASET INFO
    # ======================================

    print(
        "\nDataset Loaded Successfully!"
    )

    print(
        f"\nDataset Shape: {df.shape}"
    )

    print(
        "\nSample Data:\n"
    )

    print(df.head())

    # ======================================
    # CLEANING
    # ======================================

    df = clean_column_names(df)

    df = remove_constant_columns(df)

    df = optimize_memory(df)

    # ======================================
    # VALIDATION
    # ======================================

    validation_report = validate_dataset(
        df
    )

    print(
        "\nVALIDATION REPORT\n"
    )

    print(validation_report)

    # ======================================
    # SCHEMA DETECTION
    # ======================================

    print(
        "\nDetecting Schema...\n"
    )

    schema = detect_schema(df)

    print(schema)

    # ======================================
    # CLEAN DATA
    # ======================================

    df = clean_data(df)

    # ======================================
    # REMOVE OUTLIERS
    # ======================================

    df = remove_outliers(df)

    # ======================================
    # TIME FEATURES
    # ======================================

    if len(

        schema["datetime_columns"]

    ) > 0:

        df = build_time_features(

            df,

            schema[
                "datetime_columns"
            ]
        )

    # ======================================
    # FEATURE ENGINEERING
    # ======================================

    df = build_features(df)

    # ======================================
    # DECISION ENGINE
    # ======================================

    df = generate_decision(df)

    # ======================================
    # AVAILABLE COLUMNS
    # ======================================

    print(
        "\nAvailable Columns:\n"
    )

    for index, column in enumerate(
        df.columns
    ):

        print(
            f"{index + 1}. {column}"
        )

    # ======================================
    # TARGET COLUMN
    # ======================================

    target_column = input(
        "\nEnter target column: "
    )

    # ======================================
    # VALIDATE TARGET
    # ======================================

    if target_column not in df.columns:

        print(
            "\nInvalid target column!"
        )

        return

    # ======================================
    # TARGET TYPE DETECTION
    # ======================================

    original_target_dtype = str(

        df[target_column].dtype
    )

    original_unique_values = (

        df[target_column].nunique()
    )

    is_classification_target = False

    if (

        original_target_dtype == "object"

        or

        original_unique_values <= 20
    ):

        is_classification_target = True

    # ======================================
    # ENCODING
    # ======================================

    df, encoders = encode_features(

    df,

    exclude_columns=[
        target_column])

    # ======================================
    # SCALING
    # ======================================

    df, scaler = scale_features(

        df,

        exclude_columns=[
            target_column
        ]
    )

    # ======================================
    # SAVE PROCESSED DATASET
    # ======================================

    os.makedirs(

        "data/processed",

        exist_ok=True
    )

    processed_dataset_path = (

        "data/processed/"
        "processed_dataset.csv"
    )

    df.to_csv(

        processed_dataset_path,

        index=False
    )

    print(
        "\nProcessed Dataset Saved!"
    )

    print(
        processed_dataset_path
    )

    # ======================================
    # DATETIME COLUMN
    # ======================================

    datetime_column = None

    if len(

        schema["datetime_columns"]

    ) > 0:

        datetime_column = (

            schema[
                "datetime_columns"
            ][0]
        )

    # ======================================
    # FORECASTING MODE
    # ======================================

    if (

        datetime_column is not None

        and

        not is_classification_target
    ):

        print(
            "\nTIME-SERIES FORECASTING MODE"
        )

        (
            model,

            model_name,

            y_test,

            predictions,

            future_forecast

        ) = train_forecasting_model(

            df,

            target_column,

            datetime_column
        )

        problem_type = (
            "forecasting"
        )

    # ======================================
    # STANDARD ML MODE
    # ======================================

    else:

        print(
            "\nSTANDARD ML MODE"
        )

        (
            model,

            y_test,

            predictions,

            problem_type,

            model_name

        ) = train_model(

            df,

            target_column
        )

        future_forecast = None

    # ======================================
    # PREDICTIONS
    # ======================================

    prediction_df = pd.DataFrame({

        "Actual":
            y_test,

        "Predicted":
            predictions
    })

    print(
        "\nPREDICTION RESULTS\n"
    )

    print(
        prediction_df.head()
    )

    # ======================================
    # SAVE PREDICTIONS
    # ======================================

    os.makedirs(

        "data/predictions",

        exist_ok=True
    )

    prediction_path = (

        "data/predictions/"
        "predictions.csv"
    )

    prediction_df.to_csv(

        prediction_path,

        index=False
    )

    print(
        "\nPredictions Saved!"
    )

    print(prediction_path)

    # ======================================
    # FUTURE FORECAST
    # ======================================

    if future_forecast is not None:

        future_forecast_df = pd.DataFrame({

            "Future_Forecast":
                future_forecast
        })

        future_forecast_path = (

            "data/predictions/"
            "future_forecast.csv"
        )

        future_forecast_df.to_csv(

            future_forecast_path,

            index=False
        )

        print(
            "\nFuture Forecast Generated!"
        )

        print(
            future_forecast_df.head()
        )

        print(
            future_forecast_path
        )

    # ======================================
    # FINAL MESSAGE
    # ======================================

    print(
        "\n====================================="
    )

    print(
        "INDUSTRIAL AI PIPELINE COMPLETED!"
    )

    print(
        "=====================================\n"
    )

# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    main()