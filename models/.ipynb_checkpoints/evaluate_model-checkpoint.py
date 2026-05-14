from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import numpy as np

def evaluate_model(
    y_test,
    predictions,
    problem_type
):

    print("\nMODEL EVALUATION\n")

    if problem_type == "classification":

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(f"Accuracy: {accuracy}")

        print("\nClassification Report:\n")

        print(
            classification_report(
                y_test,
                predictions
            )
        )

        metrics = {
            "accuracy": accuracy
        }

    else:

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2 Score: {r2}")

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2
        }

    return metrics