import pandas as pd

def detect_schema(df):

    print("\nDetecting Schema...\n")

    # Numeric columns
    numeric_columns = df.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    # Categorical columns
    categorical_columns = df.select_dtypes(
        include=['object']
    ).columns.tolist()

    datetime_columns = []

    # Check only possible datetime columns
    possible_date_keywords = [
        "date",
        "time",
        "year",
        "month",
        "day"
    ]

    for col in categorical_columns:

        # Convert column name to lowercase
        col_lower = col.lower()

        # Check if column name suggests datetime
        if any(
            keyword in col_lower
            for keyword in possible_date_keywords
        ):

            try:

                pd.to_datetime(
                    df[col],
                    errors='raise'
                )

                datetime_columns.append(col)

            except:
                pass

    schema = {

        "numeric_columns":
            numeric_columns,

        "categorical_columns":
            categorical_columns,

        "datetime_columns":
            datetime_columns
    }

    print(schema)

    return schema