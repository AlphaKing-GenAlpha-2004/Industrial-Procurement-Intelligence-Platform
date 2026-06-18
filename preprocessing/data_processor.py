# =========================================================
# data_processor.py
# FINAL ENTERPRISE DATA PROCESSOR
# GENERALIZABLE AI TARGET DETECTION ENGINE
# =========================================================

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# IMPORTS
# =========================================================

import numpy as np
import pandas as pd

from sklearn.preprocessing import (

    LabelEncoder,

    StandardScaler
)

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

def clean_column_names(df):

    """
    Enterprise column normalization
    """

    df = df.copy()

    cleaned_columns = []

    for col in df.columns:

        clean_col = str(col)

        clean_col = clean_col.strip()

        clean_col = clean_col.lower()

        clean_col = clean_col.replace(" ", "_")

        clean_col = clean_col.replace("-", "_")

        clean_col = clean_col.replace("/", "_")

        clean_col = clean_col.replace(".", "_")

        clean_col = clean_col.replace("(", "")

        clean_col = clean_col.replace(")", "")

        clean_col = clean_col.replace("%", "percent")

        clean_col = clean_col.replace("&", "and")

        # Remove duplicate underscores

        while "__" in clean_col:

            clean_col = clean_col.replace(
                "__",
                "_"
            )

        clean_col = clean_col.strip("_")

        cleaned_columns.append(
            clean_col
        )

    # =====================================================
    # HANDLE DUPLICATE COLUMN NAMES
    # =====================================================

    final_columns = []

    column_counter = {}

    for col in cleaned_columns:

        if col not in column_counter:

            column_counter[col] = 0

            final_columns.append(col)

        else:

            column_counter[col] += 1

            final_columns.append(

                f"{col}_{column_counter[col]}"
            )

    df.columns = final_columns

    return df

def format_display_columns(df):

    df.columns = [

        str(col)
        .replace("_", " ")
        .title()

        for col in df.columns
    ]

    return df

# =========================================================
# REMOVE DUPLICATES
# =========================================================

def remove_duplicates(df):

    """
    Safe duplicate removal
    """

    df = df.copy()

    total_rows = len(df)

    duplicate_rows = df.duplicated().sum()

    print("\n==============================")
    print("DUPLICATE ANALYSIS")
    print("==============================")
    print(f"Total Rows      : {total_rows}")
    print(f"Duplicate Rows  : {duplicate_rows}")
    print("==============================")

    # Prevent accidental destruction of dataset

    duplicate_ratio = duplicate_rows / max(
        total_rows,
        1
    )

    if duplicate_ratio > 0.25:

        print(
            "WARNING: Excessive duplicates detected."
        )

        print(
            "Duplicate removal skipped."
        )

        return df.reset_index(
            drop=True
        )

    df = df.drop_duplicates()

    print(
        f"Rows After Removal: {len(df)}"
    )

    print("==============================\n")

    return df.reset_index(
        drop=True
    )


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

def handle_missing_values(df):

    """
    Enterprise missing value handler

    - No row deletion
    - No column deletion
    - Handles Inf values
    - Preserves dataset size
    """

    df = df.copy()

    original_rows = len(df)

    original_columns = len(df.columns)

    print("\n==============================")
    print("MISSING VALUE PROCESSING")
    print("==============================")

    total_missing_before = (

        df.isnull()

        .sum()

        .sum()
    )

    print(
        f"Missing Before: {total_missing_before}"
    )

    for col in df.columns:

        # =============================================
        # NUMERIC COLUMNS
        # =============================================

        if pd.api.types.is_numeric_dtype(
            df[col]
        ):

            df[col] = df[col].replace(

                [np.inf, -np.inf],

                np.nan
            )

            missing_count = (

                df[col]

                .isnull()

                .sum()
            )

            if missing_count == 0:

                continue

            if df[col].notna().sum() == 0:

                df[col] = 0

            else:

                median_value = (

                    df[col]

                    .median()
                )

                df[col] = (

                    df[col]

                    .fillna(
                        median_value
                    )
                )

        # =============================================
        # CATEGORICAL COLUMNS
        # =============================================

        else:

            df[col] = (

                df[col]

                .replace(

                    [
                        "nan",
                        "None",
                        "",
                        "NULL",
                        "null"
                    ],

                    np.nan
                )
            )

            missing_count = (

                df[col]

                .isnull()

                .sum()
            )

            if missing_count == 0:

                continue

            mode_values = (

                df[col]

                .mode()
            )

            fill_value = (

                mode_values.iloc[0]

                if len(mode_values) > 0

                else "unknown"
            )

            df[col] = (

                df[col]

                .fillna(
                    fill_value
                )
            )

    total_missing_after = (

        df.isnull()

        .sum()

        .sum()
    )

    print(
        f"Missing After : {total_missing_after}"
    )

    print(
        f"Rows Before   : {original_rows}"
    )

    print(
        f"Rows After    : {len(df)}"
    )

    print(
        f"Columns Before: {original_columns}"
    )

    print(
        f"Columns After : {len(df.columns)}"
    )

    print("==============================\n")

    return df

# =========================================================
# ENTERPRISE TARGET DETECTOR
# =========================================================

def detect_enterprise_targets(df):

    if df is None or df.empty:

        return {

            "targets": {},

            "confidence": {},

            "candidates": {}
        }

    target_library = {

        "demand_target": [

            "demand",
            "forecast",
            "consumption",
            "requirement",
            "usage",
            "sales",
            "orders",
            "shipment",
            "volume",
            "quantity",
            "units"
        ],

        "cost_target": [

            "cost",
            "price",
            "expense",
            "budget",
            "spend",
            "procurement",
            "purchase",
            "revenue",
            "profit",
            "margin"
        ],

        "safety_target": [

            "risk",
            "safety",
            "quality",
            "inspection",
            "compliance",
            "failure",
            "hazard",
            "incident",
            "defect"
        ],

        "inhouse_cost_target": [

        "inhouse cost",
        "in-house cost",
        "internal cost",
        "manufacturing cost",
        "production cost",
        "make cost",
        "factory cost"],

        "inhouse_capacity_target": [

        "inhouse capacity",
        "in-house capacity",
        "internal capacity",
        "manufacturing capacity",
        "production capacity",
        "plant capacity",
        "factory capacity"],

        "inhouse_safety_target": [

        "inhouse safety",
        "in-house safety",
        "internal safety",
        "plant safety",
        "factory safety",
        "manufacturing safety",
        "workplace safety"]
    }

    excluded_patterns = {

        "id",
        "uuid",
        "serial",
        "index"
    }

    results = {

        "targets": {},

        "confidence": {},

        "candidates": {}
    }

    used_columns = set()

    # =====================================================
    # SCORER
    # =====================================================

    def score_column(

        column_name,
        series,
        target_type

    ):

        score = 0

        col = str(
            column_name
        ).lower()

        # =============================================
        # EXCLUDE IDS
        # =============================================

        if any(

            pattern in col

            for pattern

            in excluded_patterns

        ):

            return -999

        # =============================================
        # KEYWORD SCORE
        # =============================================

        keywords = target_library[
            target_type
        ]

        for keyword in keywords:

            if col == keyword:

                score += 150

            elif keyword in col:

                score += 50

        # =============================================
        # DATA TYPE SCORE
        # =============================================

        is_numeric = (

            pd.api.types.is_numeric_dtype(
                series
            )
        )

        unique_count = (

            series.nunique()
        )

        unique_ratio = (

            unique_count

            /

            max(
                len(series),
                1
            )
        )

        # =============================================
        # DEMAND
        # =============================================

        if target_type == "demand_target":

            if is_numeric:

                score += 75

            if unique_count > 20:

                score += 30

            if unique_ratio > 0.05:

                score += 15

        # =============================================
        # COST
        # =============================================

        elif target_type == "cost_target":

            if is_numeric:

                score += 75

            if unique_count > 20:

                score += 30

            if unique_ratio > 0.10:

                score += 15

        # =============================================
        # SAFETY
        # =============================================

        elif target_type == "safety_target":

            if not is_numeric:

                score += 75

            if 2 <= unique_count <= 20:

                score += 30

        # =============================================
        # INHOUSE COST
        # =============================================

        elif target_type == "inhouse_cost_target":

            if is_numeric:

                score += 100

            if unique_count > 20:

                score += 25

        # =============================================
        # INHOUSE CAPACITY
        # =============================================

        elif target_type == "inhouse_capacity_target":

            if is_numeric:

                score += 100

            if unique_count > 20:

                score += 25

        # =============================================
        # INHOUSE SAFETY
        # =============================================

        elif target_type == "inhouse_safety_target":

            if is_numeric:

                score += 75

            else:

                score += 50

            if 2 <= unique_count <= 20:

                score += 25

            try:

                values = set(

                    series
                    .astype(str)
                    .str.lower()
                    .unique()
                )

                safety_words = {

                    "safe",
                    "unsafe",
                    "low",
                    "medium",
                    "high",
                    "pass",
                    "fail",
                    "critical"
                }

                score += (

                    len(
                        values &
                        safety_words
                    ) * 20
                )

            except:

                pass

        return score

    # =====================================================
    # DETECT TARGETS
    # =====================================================

    for target_type in target_library:

        column_scores = []

        for col in df.columns:

            if col in used_columns:

                continue

            score = score_column(

                col,

                df[col],

                target_type
            )

            column_scores.append({

                "column": col,

                "score": score
            })

        column_scores = sorted(

            column_scores,

            key=lambda x: x["score"],

            reverse=True
        )

        if len(column_scores) == 0:

            continue

        best = column_scores[0]

        results["targets"][
            target_type
        ] = best["column"]

        used_columns.add(
            best["column"]
        )

        # =============================================
        # CONFIDENCE
        # =============================================

        confidence = min(

            100,

            max(

                0,

                round(

                    best["score"]

                    /

                    3
                )
            )
        )

        results["confidence"][
            target_type
        ] = confidence

        results["candidates"][
            target_type
        ] = column_scores[:5]

    # =====================================================
    # DEBUG
    # =====================================================

    print("\n================================")
    print("TARGET DETECTION RESULTS")
    print("================================")

    for target_type in results["targets"]:

        print(

            target_type,

            "->",

            results["targets"][
                target_type
            ],

            "| Confidence:",

            results["confidence"][
                target_type
            ]
        )

    print("================================\n")

    return results

# =========================================================
# ENTERPRISE TASK TYPE DETECTOR
# =========================================================

def detect_task_types(df, targets):

    """
    Detect regression vs classification targets
    safely even if targets contains metadata.

    Returns:
        {
            demand_target: regression/classification,
            cost_target: regression/classification,
            safety_target: regression/classification
        }
    """

    task_types = {}

    # =====================================================
    # SAFE TARGET EXTRACTION
    # =====================================================

    target_mapping = {

        "demand_target":
        targets.get("demand_target"),

        "cost_target":
        targets.get("cost_target"),

        "safety_target":
        targets.get("safety_target")
    }

    # =====================================================
    # TASK DETECTION
    # =====================================================

    for target_name, target_column in target_mapping.items():

        # ================================================
        # INVALID TARGET TYPE
        # ================================================

        if not isinstance(
            target_column,
            str
        ):

            task_types[target_name] = None

            continue

        # ================================================
        # COLUMN NOT FOUND
        # ================================================

        if target_column not in df.columns:

            task_types[target_name] = None

            continue

        series = df[
            target_column
        ].dropna()

        if len(series) == 0:

            task_types[target_name] = None

            continue

        unique_count = series.nunique()

        row_count = len(series)

        unique_ratio = (

            unique_count

            /

            max(row_count, 1)
        )

        # ================================================
        # CATEGORICAL TARGET
        # ================================================

        if (

            pd.api.types.is_object_dtype(series)

            or

            pd.api.types.is_categorical_dtype(series)

            or

            pd.api.types.is_bool_dtype(series)

        ):

            task_types[target_name] = (

                "classification"
            )

            continue

        # ================================================
        # NUMERIC TARGET
        # ================================================

        if pd.api.types.is_numeric_dtype(series):

            # Binary classes

            if unique_count == 2:

                task_types[target_name] = (

                    "classification"
                )

                continue

            # Decimal analysis

            try:

                numeric_series = pd.to_numeric(

                    series,

                    errors="coerce"
                )

                decimal_ratio = (

                    (

                        numeric_series % 1

                    ) != 0

                ).mean()

            except Exception:

                decimal_ratio = 0

            # Regression indicators

            if (

                unique_count > 25

                or

                unique_ratio > 0.05

                or

                decimal_ratio > 0.20

            ):

                task_types[target_name] = (

                    "regression"
                )

            else:

                task_types[target_name] = (

                    "classification"
                )

            continue

        # ================================================
        # FALLBACK
        # ================================================

        task_types[target_name] = (

            "classification"
        )

    # =====================================================
    # DEBUG OUTPUT
    # =====================================================

    print("\n================================")
    print("TASK TYPE DETECTION")
    print("================================")

    for target_name, target_column in target_mapping.items():

        task_type = task_types.get(
            target_name
        )

        if (

            isinstance(
                target_column,
                str
            )

            and

            target_column in df.columns

        ):

            print(

                f"{target_name}"

                f" -> {target_column}"

                f" | Unique={df[target_column].nunique()}"

                f" | Task={task_type}"

            )

    print("================================\n")

    return task_types

# =========================================================
# ENCODE CATEGORICAL FEATURES
# =========================================================

def encode_categorical_columns(

    df,

    exclude_columns=None

):

    """
    Encode feature columns only.

    Target columns remain untouched.
    """

    df = df.copy()

    if exclude_columns is None:

        exclude_columns = []

    label_encoders = {}

    encoded_columns = []

    for col in df.columns:

        # =============================================
        # SKIP TARGETS
        # =============================================

        if col in exclude_columns:

            continue

        # =============================================
        # ENCODE NON-NUMERIC
        # =============================================

        if not pd.api.types.is_numeric_dtype(

            df[col]

        ):

            encoder = LabelEncoder()

            values = (

                df[col]

                .astype(str)

                .fillna("Unknown")
            )

            df[col] = (

                encoder.fit_transform(

                    values
                )
            )

            label_encoders[
                col
            ] = encoder

            encoded_columns.append(
                col
            )

    print("\n==============================")

    print(
        "CATEGORICAL ENCODING"
    )

    print("==============================")

    print(
        "Encoded Columns:"
    )

    print(
        encoded_columns
    )

    print("==============================\n")

    return df, label_encoders


# =========================================================
# FEATURE SCALING
# =========================================================

def scale_features(

    df,

    exclude_columns=None

):

    """
    Scale feature columns only.

    Target columns are excluded.
    """

    df = df.copy()

    if exclude_columns is None:

        exclude_columns = []

    scaler = StandardScaler()

    # =============================================
    # FEATURE COLUMNS
    # =============================================

    feature_columns = [

        col

        for col in df.columns

        if col not in exclude_columns
    ]

    # =============================================
    # NUMERIC FEATURES ONLY
    # =============================================

    numeric_features = [

        col

        for col in feature_columns

        if pd.api.types.is_numeric_dtype(

            df[col]

        )
    ]

    # =============================================
    # SCALE
    # =============================================

    if len(numeric_features) > 0:

        try:

            df[numeric_features] = (

                scaler.fit_transform(

                    df[numeric_features]
                )
            )

        except Exception as e:

            print(

                f"Scaling Error -> {e}"

            )

    print("\n==============================")

    print(
        "FEATURE SCALING"
    )

    print("==============================")

    print(
        f"Scaled Features: {len(numeric_features)}"
    )

    print(
        numeric_features[:20]
    )

    print("==============================\n")

    return df, scaler

# =========================================================
# ENTERPRISE ENTITY COLUMN DETECTOR
# =========================================================

def detect_entity_column(df):

    """
    Detect:

    - Part Name
    - Equipment Name
    - Product Name
    - Component Name
    - Material Name

    Returns:
        entity_column
    """

    if df is None or df.empty:

        return None

    entity_keywords = {

        "part": 150,
        "part_name": 200,

        "equipment": 180,
        "equipment_name": 220,

        "component": 170,
        "component_name": 210,

        "product": 160,
        "product_name": 200,

        "item": 150,
        "item_name": 190,

        "material": 140,
        "material_name": 180,

        "asset": 140,
        "asset_name": 180,

        "machine": 150,
        "machine_name": 190,

        "description": 80,
        "product_description": 100
    }

    id_keywords = [

        "id",
        "uuid",
        "serial",
        "code",
        "index",
        "key"
    ]

    candidate_scores = {}

    total_rows = len(df)

    for col in df.columns:

        score = 0

        col_name = str(col).lower()

        # =================================================
        # NAME SCORE
        # =================================================

        for keyword, weight in entity_keywords.items():

            if col_name == keyword:

                score += weight + 150

            elif keyword in col_name:

                score += weight

        # =================================================
        # ID COLUMN PENALTY
        # =================================================

        if any(

            word in col_name

            for word in id_keywords

        ):

            score -= 300

        # =================================================
        # DATA TYPE
        # =================================================

        if pd.api.types.is_object_dtype(df[col]):

            score += 100

        elif pd.api.types.is_numeric_dtype(df[col]):

            score -= 200

        # =================================================
        # CARDINALITY
        # =================================================

        unique_count = df[col].nunique(
            dropna=True
        )

        unique_ratio = (

            unique_count

            /

            max(total_rows, 1)
        )

        if 0.01 <= unique_ratio <= 0.90:

            score += 100

        elif unique_ratio > 0.95:

            score -= 150

        # =================================================
        # AVERAGE TEXT LENGTH
        # =================================================

        try:

            avg_length = (

                df[col]

                .dropna()

                .astype(str)

                .str.len()

                .mean()
            )

            if 3 <= avg_length <= 60:

                score += 50

        except:

            pass

        # =================================================
        # CODE-LIKE VALUE PENALTY
        # =================================================

        try:

            sample = (

                df[col]

                .dropna()

                .astype(str)

                .head(100)
            )

            code_ratio = (

                sample

                .str.match(
                    r"^[A-Z0-9_-]{3,}$",
                    na=False
                )

                .mean()
            )

            if code_ratio > 0.80:

                score -= 120

        except:

            pass

        candidate_scores[col] = score

    if not candidate_scores:

        return None

    entity_column = max(

        candidate_scores,

        key=candidate_scores.get
    )

    best_score = candidate_scores[
        entity_column
    ]

    print("\n========================")
    print("ENTITY DETECTION")
    print("========================")

    for col, score in sorted(

        candidate_scores.items(),

        key=lambda x: x[1],

        reverse=True

    )[:5]:

        print(
            f"{col} -> {score}"
        )

    print("========================")

    if best_score < 100:

        print(
            "Entity column not confidently detected"
        )

        return None

    print(
        f"Selected Entity Column: {entity_column}"
    )

    return entity_column

# =========================================================
# ENTERPRISE SUPPLIER DETECTOR
# =========================================================

def detect_supplier_column(df):

    """
    Detect supplier/vendor column
    """

    supplier_keywords = {

        "supplier": 250,
        "supplier_name": 300,

        "vendor": 250,
        "vendor_name": 300,

        "manufacturer": 220,
        "manufacturer_name": 260,

        "provider": 180,
        "provider_name": 220,

        "contractor": 180,

        "seller": 150,

        "distributor": 180
    }

    candidate_scores = {}

    total_rows = len(df)

    for col in df.columns:

        score = 0

        col_name = str(col).lower()

        # Name score

        for keyword, weight in supplier_keywords.items():

            if col_name == keyword:

                score += weight + 150

            elif keyword in col_name:

                score += weight

        # Object bonus

        if pd.api.types.is_object_dtype(

            df[col]

        ):

            score += 100

        else:

            score -= 150

        # Cardinality

        unique_count = df[col].nunique()

        unique_ratio = (

            unique_count

            /

            max(total_rows, 1)
        )

        if 0.001 <= unique_ratio <= 0.50:

            score += 80

        candidate_scores[col] = score

    if not candidate_scores:

        return None

    supplier_column = max(

        candidate_scores,

        key=candidate_scores.get
    )

    best_score = candidate_scores[
        supplier_column
    ]

    print("\n========================")
    print("SUPPLIER DETECTION")
    print("========================")

    for col, score in sorted(

        candidate_scores.items(),

        key=lambda x: x[1],

        reverse=True

    )[:5]:

        print(
            f"{col} -> {score}"
        )

    print("========================")

    if best_score < 100:

        print(
            "Supplier column not confidently detected"
        )

        return None

    print(
        f"Selected Supplier Column: {supplier_column}"
    )

    return supplier_column

# =========================================================
# MAIN PREPROCESSING PIPELINE
# =========================================================

def preprocess_dataset(raw_df):

    if raw_df is None:
        raise ValueError("Input dataframe is None")

    if raw_df.empty:
        raise ValueError("Uploaded dataset is empty")

    print("\n================================")
    print("PREPROCESSING STARTED")
    print("================================")

    df = raw_df.copy()

    # =====================================================
    # DATA CLEANING
    # =====================================================

    df = clean_column_names(df)

    df = remove_duplicates(df)

    df = format_display_columns(df)

    df = handle_missing_values(df)

    raw_business_df = df.copy()

    # =====================================================
    # ENTITY / SUPPLIER DETECTION
    # =====================================================

    entity_column = detect_entity_column(
        raw_business_df
    )

    supplier_column = detect_supplier_column(
        raw_business_df
    )

    # =====================================================
    # TARGET DETECTION
    # =====================================================

    detected_targets = detect_enterprise_targets(
        raw_business_df
    )

    targets = {

        "demand_target":
        detected_targets["targets"].get(
            "demand_target"
        ),

        "cost_target":
        detected_targets["targets"].get(
            "cost_target"
        ),

        "safety_target":
        detected_targets["targets"].get(
            "safety_target"
        ),

        "inhouse_cost_target":
        detected_targets["targets"].get(
            "inhouse_cost_target"
        ),

        "inhouse_capacity_target":
        detected_targets["targets"].get(
            "inhouse_capacity_target"
        ),

        "inhouse_safety_target":
        detected_targets["targets"].get(
            "inhouse_safety_target"
        )
    }

    target_confidence = detected_targets.get(
        "confidence",
        {}
    )

    target_candidates = detected_targets.get(
        "candidates",
        {}
    )

    # =====================================================
    # DEBUG TARGETS
    # =====================================================

    print("\nTARGET DETECTION RESULT")
    print(targets)

    # =====================================================
    # TASK DETECTION
    # =====================================================

    task_types = detect_task_types(

        raw_business_df,

        targets
    )

    # =====================================================
    # TARGET COLUMNS
    # =====================================================

    target_columns = [

        targets.get(
            "demand_target"
        ),

        targets.get(
            "cost_target"
        ),

        targets.get(
            "safety_target"
        ),

        targets.get(
        "inhouse_cost_target"),
        
        targets.get(
        "inhouse_capacity_target"),
        
        targets.get(
        "inhouse_safety_target")
    ]

    target_columns = [

        col

        for col in target_columns

        if (

            isinstance(col, str)

            and

            col in raw_business_df.columns
        )
    ]

    target_columns = list(
        dict.fromkeys(
            target_columns
        )
    )

    # =====================================================
    # FEATURE ENCODING
    # =====================================================

    processed_df, label_encoders = (

        encode_categorical_columns(

            raw_business_df.copy(),

            exclude_columns=
            target_columns
        )
    )

    # =====================================================
    # FEATURE DETECTION
    # =====================================================

    ignore_patterns = {

        "id",
        "uuid",
        "serial",
        "index"
    }

    feature_columns = []

    for col in processed_df.columns:

        if col in target_columns:
            continue

        if any(

            pattern in col.lower()

            for pattern in ignore_patterns

        ):
            continue

        feature_columns.append(col)

    # =====================================================
    # FEATURE SCALING
    # =====================================================

    processed_df, scaler = (

        scale_features(

            processed_df,

            exclude_columns=
            target_columns
        )
    )

    feature_columns = [

        col

        for col in feature_columns

        if col in processed_df.columns
    ]

    # =====================================================
    # ROW CONSISTENCY
    # =====================================================

    if len(processed_df) != len(raw_business_df):

        raise ValueError(

            f"Row Mismatch -> "

            f"Processed={len(processed_df)} "

            f"Raw={len(raw_business_df)}"
        )

    # =====================================================
    # DIAGNOSTICS
    # =====================================================

    diagnostics = {

        "rows":
        len(processed_df),

        "columns":
        len(processed_df.columns),

        "feature_count":
        len(feature_columns),

        "target_count":
        len(target_columns),

        "entity_column":
        entity_column,

        "supplier_column":
        supplier_column,

        "targets":
        targets,

        "confidence":
        target_confidence,

        "task_types":
        task_types,

        "missing_values":
        int(
            processed_df
            .isnull()
            .sum()
            .sum()
        )
    }

    print("\n================================")
    print("PREPROCESSING SUMMARY")
    print("================================")
    print("Entity:", entity_column)
    print("Supplier:", supplier_column)
    print("Targets:", targets)
    print("Task Types:", task_types)
    print("Features:", len(feature_columns))
    print("================================\n")

    return {

        "processed_df":
        processed_df,

        "raw_business_df":
        raw_business_df,

        "entity_column":
        entity_column,

        "supplier_column":
        supplier_column,

        "targets":
        targets,

        "target_confidence":
        target_confidence,

        "target_candidates":
        target_candidates,

        "task_types":
        task_types,

        "target_columns":
        target_columns,

        "feature_columns":
        feature_columns,

        "label_encoders":
        label_encoders,

        "scaler":
        scaler,

        "diagnostics":
        diagnostics
    }