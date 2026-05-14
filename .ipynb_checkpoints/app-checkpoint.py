import pandas as pd
import os

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

from utils.validation import (
    validate_data
)

from models.train_model import (
    train_model
)

# =========================
# CREATE DIRECTORIES
# =========================

os.makedirs(
    "data/processed",
    exist_ok=True
)

os.makedirs(
    "saved_models",
    exist_ok=True
)

print(
    "\nINDUSTRIAL AI SYSTEM\n"
)

# =========================
# LOAD DATASET
# =========================

file_path = input(
    "\nEnter dataset path: "
)

# CSV
if file_path.endswith(".csv"):

    df = pd.read_csv(file_path)

# EXCEL
elif file_path.endswith(".xlsx"):

    df = pd.read_excel(file_path)

else:

    raise ValueError(
        "Unsupported file format!"
    )

print(
    "\nDataset Loaded Successfully!\n"
)

print(df.head())

# =========================
# VALIDATION
# =========================

validate_data(df)

# =========================
# SCHEMA DETECTION
# =========================

schema = detect_schema(df)

# =========================
# CLEANING
# =========================

df = clean_data(df)

# =========================
# OUTLIER HANDLING
# =========================

df = remove_outliers(df)

# =========================
# FEATURE ENGINEERING
# =========================

df = build_features(df)

# =========================
# ENCODING
# =========================

df, encoders = encode_features(df)

# =========================
# SCALING
# =========================

df, scaler = scale_features(df)

# =========================
# SAVE PROCESSED DATA
# =========================

processed_path = (
    "data/processed/processed_data.csv"
)

df.to_csv(
    processed_path,
    index=False
)

print(
    f"\nProcessed dataset saved to:\n"
    f"{processed_path}"
)

# =========================
# TRAIN AI MODEL
# =========================

model = train_model(df)

print(
    "\nDAY 2 AI PIPELINE COMPLETED!"
)