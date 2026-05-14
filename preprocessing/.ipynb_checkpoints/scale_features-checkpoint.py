from sklearn.preprocessing import StandardScaler

def scale_features(df):

    print("\nScaling Features...\n")

    scaler = StandardScaler()

    numeric_columns = df.select_dtypes(
        include=['int64', 'float64']
    ).columns

    # Create COPY
    scaled_df = df.copy()

    scaled_df[numeric_columns] = (
        scaler.fit_transform(
            scaled_df[numeric_columns]
        )
    )

    print("Scaling Completed!")

    return scaled_df, scaler, numeric_columns