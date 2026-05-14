from sklearn.preprocessing import LabelEncoder

def encode_features(df):

    print("\nEncoding Features...\n")

    label_encoders = {}

    categorical_columns = df.select_dtypes(
        include=['object']
    ).columns

    for col in categorical_columns:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )

        label_encoders[col] = encoder

    print("Encoding Completed!")

    return df, label_encoders