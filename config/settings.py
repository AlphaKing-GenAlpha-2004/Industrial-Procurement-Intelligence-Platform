# =========================================================
# settings.py
# FINAL ENTERPRISE PROCUREMENT CONFIGURATION ENGINE
# PROCUREMENT INTELLIGENCE + DEMAND PREDICTION
# =========================================================

import os

# =========================================================
# ROOT DIRECTORIES
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "saved_models"
)

EXPORT_DIR = os.path.join(
    BASE_DIR,
    "exports"
)

CACHE_DIR = os.path.join(
    BASE_DIR,
    "cache"
)

# =========================================================
# CREATE DIRECTORIES
# =========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    EXPORT_DIR,
    exist_ok=True
)

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)

# =========================================================
# APPLICATION CONFIGURATION
# =========================================================

APP_CONFIG = {

    "app_title":
    "Enterprise Procurement Intelligence Platform",

    "page_title":
    "Industrial Procurement Decision Intelligence",

    "layout":
    "wide",

    "sidebar_state":
    "expanded",

    "max_display_rows":
    500,

    "max_upload_size_mb":
    500,

    "random_state":
    42,

    "enable_caching":
    True,

    "enable_model_saving":
    True,

    "enable_dark_theme":
    True,

    "enable_procurement_intelligence":
    True
}

# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_CONFIG = {

    # =====================================================
    # TRAIN TEST
    # =====================================================

    "test_size":
    0.20,

    "cross_validation":
    5,

    "random_state":
    42,

    # =====================================================
    # TREE MODELS
    # =====================================================

    "n_estimators":
    120,

    "max_depth":
    12,

    "min_samples_split":
    4,

    "min_samples_leaf":
    1,

    "max_features":
    "sqrt",

    # =====================================================
    # PERFORMANCE
    # =====================================================

    "enable_parallel_training":
    True,

    "enable_model_persistence":
    True,

    "enable_feature_importance":
    True
}

# =========================================================
# PREPROCESSING CONFIGURATION
# =========================================================

PREPROCESSING_CONFIG = {

    "remove_duplicates":
    True,

    "handle_missing_values":
    True,

    "remove_outliers":
    True,

    "scale_features":
    True,

    "encode_categorical":
    True,

    "feature_engineering":
    True,

    "drop_high_cardinality":
    True,

    "drop_constant_columns":
    True,

    "missing_numeric_strategy":
    "median",

    "missing_categorical_strategy":
    "most_frequent"
}

# =========================================================
# TARGET KEYWORDS
# =========================================================

TARGET_KEYWORDS = {

    "demand": [

        "demand",

        "sales",

        "orders",

        "quantity",

        "inventory",

        "stock",

        "consumption",

        "production",

        "units"
    ],

    "cost": [

        "cost",

        "price",

        "expense",

        "budget",

        "revenue",

        "spend",

        "profit"
    ],

    "safety": [

        "safety",

        "risk",

        "hazard",

        "failure",

        "incident",

        "defect",

        "downtime",

        "maintenance"
    ]
}

# =========================================================
# FEATURE ENGINEERING CONFIGURATION
# =========================================================

FEATURE_ENGINEERING_CONFIG = {

    "enable_inventory_turnover":
    True,

    "enable_cost_efficiency":
    True,

    "enable_machine_efficiency":
    True,

    "enable_supplier_performance":
    True,

    "enable_operational_risk":
    True,

    "enable_procurement_features":
    True
}

# =========================================================
# PROCUREMENT CONFIGURATION
# =========================================================

PROCUREMENT_CONFIG = {

    # =====================================================
    # PROCUREMENT WEIGHTS
    # =====================================================

    "demand_weight":
    0.45,

    "cost_weight":
    0.35,

    "safety_weight":
    0.20,

    # =====================================================
    # SAFETY SCORES
    # =====================================================

    "low_risk_score":
    90,

    "medium_risk_score":
    70,

    "high_risk_score":
    45,

    # =====================================================
    # PROCUREMENT THRESHOLDS
    # =====================================================

    "strategic_percentile":
    85,

    "priority_percentile":
    60,

    "moderate_percentile":
    35
}

# =========================================================
# PROCUREMENT SCORE MAP
# =========================================================

PROCUREMENT_SCORE_MAP = {

    "Strategic Procurement":
    95,

    "Priority Procurement":
    80,

    "Moderate Procurement":
    60,

    "Restricted Procurement":
    35
}

# =========================================================
# PROCUREMENT RECOMMENDATIONS
# =========================================================

PROCUREMENT_RECOMMENDATIONS = {

    "Strategic Procurement":
    "Immediate strategic procurement recommended.",

    "Priority Procurement":
    "Optimize procurement allocation and sourcing.",

    "Moderate Procurement":
    "Monitor procurement demand and inventory.",

    "Restricted Procurement":
    "Restrict procurement exposure due to elevated risk."
}

# =========================================================
# KPI CONFIGURATION
# =========================================================

KPI_CONFIG = {

    # =====================================================
    # REGRESSION
    # =====================================================

    "excellent_r2":
    0.90,

    "good_r2":
    0.75,

    "poor_r2":
    0.50,

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    "excellent_accuracy":
    0.90,

    "good_accuracy":
    0.75,

    "poor_accuracy":
    0.50,

    # =====================================================
    # PROCUREMENT HEALTH
    # =====================================================

    "excellent_health_score":
    85,

    "moderate_health_score":
    60,

    "critical_health_score":
    40
}

# =========================================================
# THEME CONFIGURATION
# =========================================================

THEME_CONFIG = {

    "primary_color":
    "#00f5a0",

    "secondary_color":
    "#00d9f5",

    "purple_color":
    "#8b5cf6",

    "danger_color":
    "#ef4444",

    "warning_color":
    "#f59e0b",

    "background_color":
    "#020617",

    "card_background":
    "rgba(255,255,255,0.04)",

    "glass_border":
    "rgba(255,255,255,0.08)",

    "text_primary":
    "#f8fafc",

    "text_secondary":
    "#94a3b8"
}

# =========================================================
# CHART CONFIGURATION
# =========================================================

CHART_CONFIG = {

    "default_height":
    500,

    "large_chart_height":
    700,

    "top_records":
    20,

    "heatmap_limit":
    1000,

    "enable_animations":
    True
}

# =========================================================
# EXPORT CONFIGURATION
# =========================================================

EXPORT_CONFIG = {

    "enable_csv_export":
    True,

    "enable_excel_export":
    True,

    "enable_report_export":
    True
}

# =========================================================
# CACHE CONFIGURATION
# =========================================================

CACHE_CONFIG = {

    "dataset_cache_ttl":
    3600,

    "model_cache_ttl":
    7200
}

# =========================================================
# VALIDATION CONFIGURATION
# =========================================================

VALIDATION_CONFIG = {

    "minimum_rows":
    20,

    "minimum_columns":
    2,

    "maximum_missing_percentage":
    80
}

# =========================================================
# MODEL REGISTRY
# =========================================================

MODEL_REGISTRY = {

    "regression_models": [

        "Linear Regression",

        "Ridge Regression",

        "ElasticNet",

        "Random Forest Regressor",

        "Gradient Boosting Regressor",

        "Extra Trees Regressor",

        "AdaBoost Regressor",

        "HistGradientBoosting",

        "XGBoost Regressor"
    ],

    "classification_models": [

        "Logistic Regression",

        "Random Forest Classifier",

        "Gradient Boosting Classifier",

        "Extra Trees Classifier",

        "AdaBoost Classifier",

        "XGBoost Classifier"
    ]
}

# =========================================================
# ENTERPRISE LABELS
# =========================================================

ENTERPRISE_LABELS = {

    "overview":
    "Enterprise Overview",

    "demand":
    "Demand Intelligence",

    "cost":
    "Cost Intelligence",

    "safety":
    "Safety Intelligence",

    "procurement":
    "Procurement Intelligence",

    "supplier":
    "Supplier Analytics",

    "models":
    "AI Model Performance",

    "dataset":
    "Dataset Intelligence"
}