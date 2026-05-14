import pandas as pd
import numpy as np

def clean_data(df):

    print("\nCleaning Dataset...\n")

    # Remove duplicates
    df = df.drop_duplicates()

    # Replace infinite values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Detect numeric columns
    numeric_columns = df.select_dtypes(
        include=['int64', 'float64']
    ).columns

    # Detect categorical columns
    categorical_columns = df.select_dtypes(
        include=['object']
    ).columns

    # Fill numeric missing values
    for col in numeric_columns:

        median_value = df[col].median()

        df[col] = df[col].fillna(
            median_value
        )

    # Fill categorical missing values
    for col in categorical_columns:

        mode_value = df[col].mode()[0]

        df[col] = df[col].fillna(
            mode_value
        )

    print("Cleaning Completed!")

    return df