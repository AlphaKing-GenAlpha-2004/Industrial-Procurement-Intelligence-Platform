from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from pandas.api.types import is_numeric_dtype

import joblib
import json
import os

from models.model_selector import (
    select_model
)

from models.evaluate_model import (
    evaluate_model
)


def train_model(df, target_column):

    print("\nAI MODEL TRAINING PIPELINE\n")

    # =========================
    # VALIDATE TARGET
    # =========================

    if target_column not in df.columns:

        raise ValueError(
            "Invalid target column!"
        )

    # =========================
    # FEATURE / TARGET SPLIT
    # =========================

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    # Preserve original target
    original_y = y.copy()

    # =========================
    # PROBLEM TYPE DETECTION
    # =========================

    if original_y.dtype == "object":

        problem_type = "classification"

    elif (
        is_numeric_dtype(original_y)
        and original_y.nunique() <= 15
    ):

        problem_type = "classification"

    else:

        problem_type = "regression"

    print(
        f"\nDetected Problem Type: "
        f"{problem_type}"
    )

    # =========================
    # SCALE FEATURES ONLY
    # =========================

    scaler = StandardScaler()

    numeric_columns = X.select_dtypes(
        include=['int64', 'float64']
    ).columns

    X[numeric_columns] = scaler.fit_transform(
        X[numeric_columns]
    )

    # =========================
    # TRAIN TEST SPLIT
    # =========================

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    # =========================
    # MODEL SELECTION
    # =========================

    model = select_model(
        problem_type
    )

    # =========================
    # MODEL TRAINING
    # =========================

    print("\nTraining Model...\n")

    model.fit(
        X_train,
        y_train
    )

    print("Training Completed!")

    # =========================
    # PREDICTIONS
    # =========================

    predictions = model.predict(
        X_test
    )

    # =========================
    # EVALUATION
    # =========================

    metrics = evaluate_model(
        y_test,
        predictions,
        problem_type
    )

    # =========================
    # SAVE MODEL
    # =========================

    os.makedirs(
        "saved_models",
        exist_ok=True
    )

    model_path = (
        "saved_models/trained_model.pkl"
    )

    scaler_path = (
        "saved_models/scaler.pkl"
    )

    metadata_path = (
        "saved_models/model_metadata.json"
    )

    # Save model
    joblib.dump(
        model,
        model_path
    )

    # Save scaler
    joblib.dump(
        scaler,
        scaler_path
    )

    # Save metadata
    metadata = {

        "target_column":
            target_column,

        "problem_type":
            problem_type,

        "metrics":
            metrics
    }

    with open(
        metadata_path,
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        f"\nModel saved to:\n{model_path}"
    )

    print(
        f"\nScaler saved to:\n{scaler_path}"
    )

    print(
        f"\nMetadata saved to:\n"
        f"{metadata_path}"
    )

    return model,y_test,predictions,problem_type