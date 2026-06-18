# =========================================================
# model_engine.py
# FINAL OPTIMIZED ENTERPRISE MODEL ENGINE
# FAST + STABLE + PROCUREMENT INTELLIGENCE READY
# =========================================================

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# IMPORTS
# =========================================================

import numpy as np
import pandas as pd
import traceback

# =========================================================
# SKLEARN
# =========================================================

from sklearn.model_selection import (

    train_test_split,

    cross_val_score
)

from sklearn.metrics import (

    r2_score,

    mean_absolute_error,

    mean_squared_error,

    accuracy_score,

    balanced_accuracy_score,

    precision_score,

    recall_score,

    f1_score
)

from sklearn.preprocessing import (
    StandardScaler,

    LabelEncoder
)

# =========================================================
# REGRESSION MODELS
# =========================================================

from sklearn.linear_model import (

    LinearRegression,

    Ridge,

    ElasticNet
)

from sklearn.ensemble import (

    RandomForestRegressor,

    ExtraTreesRegressor,

    GradientBoostingRegressor
)

# =========================================================
# CLASSIFICATION MODELS
# =========================================================

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.ensemble import (

    RandomForestClassifier,

    ExtraTreesClassifier,

    GradientBoostingClassifier
)

# =========================================================
# XGBOOST
# =========================================================

try:

    from xgboost import (

        XGBRegressor,

        XGBClassifier
    )

    XGBOOST_AVAILABLE = True

except:

    XGBOOST_AVAILABLE = False

# =========================================================
# FEATURE IMPORTANCE EXTRACTION
# =========================================================

def extract_feature_importance(

    model,
    feature_columns

):

    """
    Enterprise Feature Importance Extractor

    Supports:
    - Random Forest
    - Extra Trees
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - Linear Regression
    - Ridge
    - Lasso
    - ElasticNet
    """

    try:

        # =============================================
        # TREE MODELS
        # =============================================

        if hasattr(

            model,

            "feature_importances_"
        ):

            importance = np.asarray(

                model.feature_importances_

            ).astype(float)

        # =============================================
        # LINEAR MODELS
        # =============================================

        elif hasattr(

            model,

            "coef_"
        ):

            importance = np.abs(

                np.ravel(

                    model.coef_
                )
            ).astype(float)

        # =============================================
        # UNSUPPORTED MODEL
        # =============================================

        else:

            print(

                f"Feature Importance Not Supported -> "
                f"{type(model).__name__}"
            )

            return pd.DataFrame()

        # =============================================
        # VALIDATION
        # =============================================

        if len(importance) == 0:

            return pd.DataFrame()

        if len(importance) != len(feature_columns):

            print(

                f"Importance Length Mismatch -> "
                f"Importance={len(importance)} "
                f"Features={len(feature_columns)}"
            )

            min_length = min(

                len(importance),

                len(feature_columns)
            )

            importance = importance[
                :min_length
            ]

            feature_columns = feature_columns[
                :min_length
            ]

        # =============================================
        # NEGATIVE PROTECTION
        # =============================================

        importance = np.abs(
            importance
        )

        total_importance = float(

            importance.sum()
        )

        if total_importance <= 0:

            importance_percentage = np.zeros(

                len(importance)
            )

        else:

            importance_percentage = (

                importance

                / total_importance

            ) * 100

        # =============================================
        # DATAFRAME
        # =============================================

        feature_importance_df = pd.DataFrame({

            "Feature":
            feature_columns,

            "Importance":
            np.round(

                importance_percentage,

                4
            )
        })

        feature_importance_df = (

            feature_importance_df

            .sort_values(

                by="Importance",

                ascending=False
            )

            .reset_index(
                drop=True
            )
        )

        # =============================================
        # REMOVE ZERO IMPORTANCE
        # =============================================

        feature_importance_df = (

            feature_importance_df[

                feature_importance_df[
                    "Importance"
                ] > 0

            ]
        )

        # =============================================
        # TOP FEATURES
        # =============================================

        feature_importance_df = (

            feature_importance_df

            .head(20)
        )

        print(

            f"Feature Importance Generated -> "
            f"{len(feature_importance_df)} Features"
        )

        return feature_importance_df

    except Exception as e:

        print(

            f"Feature Importance Failure -> {e}"
        )

        return pd.DataFrame()

# =========================================================
# REGRESSION MODELS
# =========================================================

def get_regression_models():

    models = {

        "Linear Regression":
        LinearRegression(),

        "Ridge":
        Ridge(
            alpha=1.0
        ),

        "Random Forest":
        RandomForestRegressor(

            n_estimators=200,

            max_depth=20,

            min_samples_leaf=2,

            random_state=42,

            n_jobs=-1
        ),

        "Extra Trees":
        ExtraTreesRegressor(

            n_estimators=200,

            max_depth=20,

            min_samples_leaf=2,

            random_state=42,

            n_jobs=-1
        )
    }

    if XGBOOST_AVAILABLE:

        models["XGBoost"] = XGBRegressor(

            n_estimators=200,

            max_depth=6,

            learning_rate=0.05,

            subsample=0.9,

            colsample_bytree=0.9,

            random_state=42,

            verbosity=0,

            objective="reg:squarederror"
        )

    return models


# =========================================================
# CLASSIFICATION MODELS
# =========================================================

def get_classification_models():

    models = {

        "Logistic Regression":
        LogisticRegression(

            max_iter=2000,

            class_weight="balanced",

            random_state=42
        ),

        "Random Forest":
        RandomForestClassifier(

            n_estimators=200,

            max_depth=20,

            min_samples_leaf=2,

            class_weight="balanced",

            random_state=42,

            n_jobs=-1
        ),

        "Extra Trees":
        ExtraTreesClassifier(

            n_estimators=200,

            max_depth=20,

            min_samples_leaf=2,

            class_weight="balanced",

            random_state=42,

            n_jobs=-1
        ),

        "Gradient Boosting":
        GradientBoostingClassifier(

            n_estimators=150,

            learning_rate=0.05,

            max_depth=5,

            random_state=42
        )
    }

    if XGBOOST_AVAILABLE:

        models["XGBoost"] = XGBClassifier(

            n_estimators=200,

            max_depth=6,

            learning_rate=0.05,

            subsample=0.9,

            colsample_bytree=0.9,

            random_state=42,

            verbosity=0,

            eval_metric="mlogloss"
        )

    return models

# =========================================================
# ENTERPRISE REGRESSION TRAINER
# =========================================================

def train_regression_pipeline(

    X,
    y,
    feature_columns

):

    # =====================================================
    # FEATURE CLEANING
    # =====================================================

    X = pd.DataFrame(X).copy()

    X = X.replace(

        [np.inf, -np.inf],

        np.nan
    )

    X = X.fillna(0)

    X = X.reset_index(
        drop=True
    )

    # =====================================================
    # TARGET CLEANING
    # =====================================================

    y = pd.Series(y).copy()

    y = pd.to_numeric(

        y,

        errors="coerce"
    )

    if y.isnull().all():

        print(
            "Regression Failure -> Empty Target"
        )

        return None

    y = y.fillna(

        y.median()
    )

    y = y.reset_index(
        drop=True
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    if len(X) != len(y):

        raise ValueError(

            f"Length Mismatch -> "

            f"X={len(X)} "

            f"Y={len(y)}"
        )

    if len(X) < 20:

        print(
            "Regression Failure -> Dataset Too Small"
        )

        return None

    if y.nunique() <= 1:

        print(
            "Regression Failure -> Constant Target"
        )

        return None

    # =====================================================
    # DEBUG
    # =====================================================

    print("\n================================")

    print(
        "REGRESSION TRAINER"
    )

    print("================================")

    print(
        "Rows:",
        len(y)
    )

    print(
        "Features:",
        X.shape[1]
    )

    print(
        "Target Unique:",
        y.nunique()
    )

    print(
        "Target Min:",
        y.min()
    )

    print(
        "Target Max:",
        y.max()
    )

    print("================================")

    # =====================================================
    # TARGET SCALING
    # =====================================================

    target_scaler = StandardScaler()

    y_scaled = (

        target_scaler.fit_transform(

            y.values.reshape(-1, 1)

        ).ravel()
    )

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    try:

        X_train, X_test, y_train, y_test = (

            train_test_split(

                X,

                y_scaled,

                test_size=0.20,

                random_state=42
            )
        )

    except Exception as e:

        print(
            f"Split Failure -> {e}"
        )

        return None

    # =====================================================
    # MODELS
    # =====================================================

    models = get_regression_models()

    best_model = None

    best_model_name = None

    best_metrics = None

    best_predictions = None

    best_feature_importance = None

    best_score = -999999

    all_model_scores = []

    # =====================================================
    # TRAINING LOOP
    # =====================================================

    for model_name, model in models.items():

        try:

            print(
                f"\nTraining -> {model_name}"
            )

            model.fit(

                X_train,

                y_train
            )

            train_pred_scaled = (

                model.predict(
                    X_train
                )
            )

            test_pred_scaled = (

                model.predict(
                    X_test
                )
            )

            full_pred_scaled = (

                model.predict(X)
            )

            if np.isnan(

                full_pred_scaled

            ).any():

                raise ValueError(
                    "NaN Predictions"
                )

            # =============================================
            # INVERSE SCALE
            # =============================================

            train_pred = (

                target_scaler.inverse_transform(

                    train_pred_scaled.reshape(-1, 1)

                ).ravel()
            )

            test_pred = (

                target_scaler.inverse_transform(

                    test_pred_scaled.reshape(-1, 1)

                ).ravel()
            )

            full_pred = (

                target_scaler.inverse_transform(

                    full_pred_scaled.reshape(-1, 1)

                ).ravel()
            )

            y_train_actual = (

                target_scaler.inverse_transform(

                    y_train.reshape(-1, 1)

                ).ravel()
            )

            y_test_actual = (

                target_scaler.inverse_transform(

                    y_test.reshape(-1, 1)

                ).ravel()
            )

            # =============================================
            # METRICS
            # =============================================

            train_r2 = r2_score(

                y_train_actual,

                train_pred
            )

            test_r2 = r2_score(

                y_test_actual,

                test_pred
            )

            mae = mean_absolute_error(

                y_test_actual,

                test_pred
            )

            rmse = np.sqrt(

                mean_squared_error(

                    y_test_actual,

                    test_pred
                )
            )

            safe_actual = np.where(

                np.abs(
                    y_test_actual
                ) < 1e-6,

                1,

                y_test_actual
            )

            mape = np.mean(

                np.abs(

                    (

                        y_test_actual

                        -

                        test_pred

                    )

                    /

                    safe_actual

                )

            ) * 100

            # =============================================
            # CROSS VALIDATION
            # =============================================

            try:

                cv_rows = min(
                    len(X),
                    5000
                )

                if len(X) > cv_rows:

                    sample_idx = (

                        X.sample(

                            cv_rows,

                            random_state=42

                        ).index
                    )

                    X_cv = X.loc[
                        sample_idx
                    ]

                    y_cv = y.loc[
                        sample_idx
                    ]

                else:

                    X_cv = X

                    y_cv = y

                cv_score = np.mean(

                    cross_val_score(

                        model,

                        X_cv,

                        y_cv,

                        cv=5,

                        scoring="r2"
                    )
                )

            except Exception:

                cv_score = test_r2

            # =============================================
            # ENTERPRISE SCORE
            # =============================================

            overfit_gap = abs(

                train_r2

                -

                test_r2
            )

            enterprise_score = (

                (test_r2 * 50)

                +

                (cv_score * 25)

                +

                ((1 / (1 + mae)) * 10)

                +

                ((1 / (1 + rmse)) * 10)

                +

                ((1 / (1 + (mape / 100))) * 5)

                -

                (overfit_gap * 15)
            )

            metrics = {

                "Train R2":
                round(train_r2, 4),

                "Test R2":
                round(test_r2, 4),

                "MAE":
                round(mae, 4),

                "RMSE":
                round(rmse, 4),

                "MAPE":
                round(mape, 2),

                "CV Score":
                round(cv_score, 4),

                "Overfit Gap":
                round(overfit_gap, 4),

                "Enterprise Score":
                round(
                    enterprise_score,
                    2
                )
            }

            all_model_scores.append({

                "Model":
                model_name,

                "Score":
                round(
                    enterprise_score,
                    2
                )
            })

            # =============================================
            # BEST MODEL
            # =============================================

            if enterprise_score > best_score:

                best_score = enterprise_score

                best_model = model

                best_model_name = model_name

                best_metrics = metrics

                best_predictions = full_pred

                try:

                    best_feature_importance = (

                        extract_feature_importance(

                            model,

                            feature_columns
                        )
                    )

                except Exception:

                    best_feature_importance = None

            print(
                f"{model_name} SUCCESS"
            )

        except Exception as e:

            print(

                f"{model_name} FAILED -> {e}"

            )

            continue

    # =====================================================
    # FAILSAFE
    # =====================================================

    if best_model is None:

        print(
            "\nALL REGRESSION MODELS FAILED"
        )

        return None

    if (

        best_feature_importance is None

        or

        len(best_feature_importance) == 0

    ):

        best_feature_importance = pd.DataFrame({

            "Feature":
            feature_columns,

            "Importance":
            np.zeros(
                len(feature_columns)
            )
        })

    # =====================================================
    # FINAL VALIDATION
    # =====================================================

    if len(best_predictions) != len(X):

        raise ValueError(

            f"Prediction Length Mismatch -> "

            f"Predictions={len(best_predictions)} "

            f"Rows={len(X)}"
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    print("\n================================")

    print(
        "REGRESSION TRAINING COMPLETE"
    )

    print(
        "Best Model:",
        best_model_name
    )

    print(
        "Rows:",
        len(X)
    )

    print(
        "Predictions:",
        len(best_predictions)
    )

    print(
        "Best Score:",
        round(
            best_score,
            2
        )
    )

    print("================================")

    # =====================================================
    # RETURN
    # =====================================================

    return {

        "model":
        best_model,

        "target_scaler":
        target_scaler,

        "best_model_name":
        best_model_name,

        "metrics":
        best_metrics,

        "full_actual":
        np.asarray(y),

        "full_predictions":
        np.asarray(
            best_predictions
        ),

        "feature_importance":
        best_feature_importance,

        "top_features":

        best_feature_importance[
            "Feature"
        ].head(10).tolist(),

        "all_model_scores":
        all_model_scores,

        "feature_columns":
        feature_columns,

        "diagnostics": {

            "rows":
            int(len(X)),

            "features":
            int(
                len(feature_columns)
            ),

            "target_unique":
            int(
                y.nunique()
            ),

            "prediction_count":
            int(
                len(best_predictions)
            ),

            "actual_count":
            int(
                len(y)
            ),

            "best_model":
            best_model_name,

            "best_score":
            round(
                best_score,
                2
            )
        }
    }
# =========================================================
# ENTERPRISE CLASSIFICATION TRAINER
# =========================================================

def train_classification_pipeline(

    X,
    y,
    feature_columns

):

    # =====================================================
    # FEATURE CLEANING
    # =====================================================

    X = pd.DataFrame(X).copy()

    X = X.replace(

        [np.inf, -np.inf],

        np.nan
    )

    X = X.fillna(0)

    X = X.reset_index(
        drop=True
    )

    # =====================================================
    # TARGET CLEANING
    # =====================================================

    y = pd.Series(y).copy()

    y = y.fillna(
        "Unknown"
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    if len(X) != len(y):

        raise ValueError(

            f"Length Mismatch -> "

            f"X={len(X)} "

            f"Y={len(y)}"
        )

    if len(X) < 20:

        print(
            "Classification Failure -> Dataset Too Small"
        )

        return None

    # =====================================================
    # LABEL ENCODING
    # =====================================================

    target_encoder = None

    if not pd.api.types.is_numeric_dtype(
        y
    ):

        target_encoder = LabelEncoder()

        y_encoded = (

            target_encoder.fit_transform(

                y.astype(str)

            )
        )

    else:

        y = pd.to_numeric(

            y,

            errors="coerce"
        )

        if y.mode().empty:

            print(
                "Classification Failure -> Invalid Target"
            )

            return None

        y = y.fillna(

            y.mode().iloc[0]
        )

        y_encoded = np.array(y)

    # =====================================================
    # CLASS ANALYSIS
    # =====================================================

    unique_classes = np.unique(
        y_encoded
    )

    class_counts = pd.Series(
        y_encoded
    ).value_counts()

    print("\n================================")

    print(
        "CLASSIFICATION TRAINER"
    )

    print("================================")

    print(
        "Rows:",
        len(y_encoded)
    )

    print(
        "Features:",
        X.shape[1]
    )

    print(
        "Classes:",
        len(unique_classes)
    )

    print(
        class_counts
    )

    print("================================")

    # =====================================================
    # CLASS VALIDATION
    # =====================================================

    if len(unique_classes) < 2:

        print(
            "Classification Failure -> Single Class"
        )

        return None

    rare_classes = (

        class_counts < 2

    ).sum()

    use_stratify = (

        class_counts.min()

        >= 2
    )

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    try:

        X_train, X_test, y_train, y_test = (

            train_test_split(

                X,

                y_encoded,

                test_size=0.20,

                random_state=42,

                stratify=

                y_encoded

                if use_stratify

                else None
            )
        )

    except Exception as e:

        print(
            f"Split Failure -> {e}"
        )

        return None

    # =====================================================
    # MODELS
    # =====================================================

    models = get_classification_models()

    best_model = None

    best_model_name = None

    best_metrics = None

    best_predictions = None

    best_feature_importance = None

    best_score = -999999

    all_model_scores = []

    # =====================================================
    # TRAINING LOOP
    # =====================================================

    for model_name, model in models.items():

        try:

            print(
                f"\nTraining -> {model_name}"
            )

            model.fit(

                X_train,

                y_train
            )

            train_pred = (

                model.predict(
                    X_train
                )
            )

            test_pred = (

                model.predict(
                    X_test
                )
            )

            full_pred = (

                model.predict(X)
            )

            # =============================================
            # METRICS
            # =============================================

            train_acc = accuracy_score(

                y_train,

                train_pred
            )

            test_acc = accuracy_score(

                y_test,

                test_pred
            )

            balanced_acc = (

                balanced_accuracy_score(

                    y_test,

                    test_pred
                )
            )

            precision = precision_score(

                y_test,

                test_pred,

                average="weighted",

                zero_division=0
            )

            recall = recall_score(

                y_test,

                test_pred,

                average="weighted",

                zero_division=0
            )

            f1 = f1_score(

                y_test,

                test_pred,

                average="weighted",

                zero_division=0
            )

            # =============================================
            # CROSS VALIDATION
            # =============================================

            try:

                cv_rows = min(
                    len(X_train),
                    5000
                )

                if len(X_train) > cv_rows:

                    sample_idx = (

                        X_train.sample(

                            cv_rows,

                            random_state=42

                        ).index
                    )

                    X_cv = X_train.loc[
                        sample_idx
                    ]

                    y_cv = pd.Series(
                        y_train
                    ).loc[
                        sample_idx
                    ]

                else:

                    X_cv = X_train

                    y_cv = y_train

                cv_score = np.mean(

                    cross_val_score(

                        model,

                        X_cv,

                        y_cv,

                        cv=5,

                        scoring=
                        "balanced_accuracy"
                    )
                )

            except Exception:

                cv_score = test_acc

            # =============================================
            # ENTERPRISE SCORE
            # =============================================

            overfit_gap = abs(

                train_acc

                -

                test_acc
            )

            enterprise_score = (

                (test_acc * 40)

                +

                (balanced_acc * 20)

                +

                (f1 * 15)

                +

                (precision * 10)

                +

                (recall * 5)

                +

                (cv_score * 20)

                -

                (overfit_gap * 20)
            )

            metrics = {

                "Train Accuracy":
                round(train_acc, 4),

                "Test Accuracy":
                round(test_acc, 4),

                "Balanced Accuracy":
                round(balanced_acc, 4),

                "Precision":
                round(precision, 4),

                "Recall":
                round(recall, 4),

                "F1 Score":
                round(f1, 4),

                "CV Score":
                round(cv_score, 4),

                "Overfit Gap":
                round(overfit_gap, 4),

                "Enterprise Score":
                round(
                    enterprise_score,
                    2
                )
            }

            all_model_scores.append({

                "Model":
                model_name,

                "Score":
                round(
                    enterprise_score,
                    2
                )
            })

            # =============================================
            # BEST MODEL
            # =============================================

            if enterprise_score > best_score:

                best_score = enterprise_score

                best_model = model

                best_model_name = model_name

                best_metrics = metrics

                best_predictions = full_pred

                try:

                    best_feature_importance = (

                        extract_feature_importance(

                            model,

                            feature_columns
                        )
                    )

                except Exception:

                    best_feature_importance = None

            print(
                f"{model_name} SUCCESS"
            )

        except Exception as e:

            print(
                f"{model_name} FAILED -> {e}"
            )

            continue

    # =====================================================
    # FAILSAFE
    # =====================================================

    if best_model is None:

        print(
            "\nALL CLASSIFICATION MODELS FAILED"
        )

        return None

    # =====================================================
    # FEATURE IMPORTANCE FALLBACK
    # =====================================================

    if (

        best_feature_importance is None

        or

        len(best_feature_importance) == 0

    ):

        best_feature_importance = pd.DataFrame({

            "Feature":
            feature_columns,

            "Importance":
            np.zeros(
                len(feature_columns)
            )
        })

    # =====================================================
    # LABEL DECODE
    # =====================================================

    full_actual = np.asarray(
        y_encoded
    )

    if target_encoder is not None:

        try:

            best_predictions = np.asarray(
                best_predictions
            ).astype(int)

            best_predictions = np.clip(

                best_predictions,

                0,

                len(
                    target_encoder.classes_
                ) - 1
            )

            full_actual = np.asarray(
                full_actual
            ).astype(int)

            full_actual = np.clip(

                full_actual,

                0,

                len(
                    target_encoder.classes_
                ) - 1
            )

            best_predictions = (

                target_encoder.inverse_transform(

                    best_predictions
                )
            )

            full_actual = (

                target_encoder.inverse_transform(

                    full_actual
                )
            )

        except Exception as e:

            print(
                f"Decode Failure -> {e}"
            )

    # =====================================================
    # FINAL VALIDATION
    # =====================================================

    if len(best_predictions) != len(X):

        raise ValueError(

            f"Prediction Length Mismatch -> "

            f"Predictions={len(best_predictions)} "

            f"Rows={len(X)}"
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    print("\n================================")

    print(
        "CLASSIFICATION TRAINING COMPLETE"
    )

    print(
        "Best Model:",
        best_model_name
    )

    print(
        "Rows:",
        len(X)
    )

    print(
        "Predictions:",
        len(best_predictions)
    )

    print(
        "Best Score:",
        round(
            best_score,
            2
        )
    )

    print("================================")

    # =====================================================
    # RETURN
    # =====================================================

    return {

        "model":
        best_model,

        "target_encoder":
        target_encoder,

        "best_model_name":
        best_model_name,

        "metrics":
        best_metrics,

        "full_actual":
        np.asarray(
            full_actual
        ),

        "full_predictions":
        np.asarray(
            best_predictions
        ),

        "feature_importance":
        best_feature_importance,

        "top_features":

        best_feature_importance[
            "Feature"
        ].head(10).tolist(),

        "all_model_scores":
        all_model_scores,

        "feature_columns":
        feature_columns,

        "diagnostics": {

            "rows":
            int(len(X)),

            "features":
            int(
                len(feature_columns)
            ),

            "classes":
            int(
                len(unique_classes)
            ),

            "rare_classes":
            int(
                rare_classes
            ),

            "prediction_count":
            int(
                len(best_predictions)
            ),

            "best_model":
            best_model_name,

            "best_score":
            round(
                best_score,
                2
            )
        }
    }

# =========================================================
# SUPPLIER COMPETITION INTELLIGENCE ENGINE
# =========================================================

def build_supplier_intelligence(
    raw_business_df,
    pipeline_output,
    entity_column,
    supplier_column,
    targets
):

    """
    Part-Level Supplier Competition Engine

    Output:
        Part Name
        Best Demand Supplier
        Best Cost Supplier
        Best Safety Supplier
        Recommended Supplier
        Procurement Score
    """

    print("ENTERED build_supplier_intelligence")

    try:

        # =================================================
        # VALIDATION
        # =================================================

        if raw_business_df is None:

            return pd.DataFrame()

        if raw_business_df.empty:

            return pd.DataFrame()

        if entity_column is None:

            return pd.DataFrame()

        if supplier_column is None:

            return pd.DataFrame()

        if entity_column not in raw_business_df.columns:

            return pd.DataFrame()

        if supplier_column not in raw_business_df.columns:

            return pd.DataFrame()

        # =================================================
        # INHOUSE TARGETS
        # =================================================

        inhouse_cost_column = targets.get("inhouse_cost_target")

        inhouse_capacity_column = targets.get("inhouse_capacity_target")

        inhouse_safety_column = targets.get("inhouse_safety_target")

        # =================================================
        # MASTER DATAFRAME
        # =================================================

        intelligence_df = raw_business_df.copy()

        intelligence_df = intelligence_df.reset_index(
            drop=True
        )

        # =================================================
        # DEMAND PREDICTIONS
        # =================================================

        demand_model = pipeline_output.get(
            "demand_model"
        )

        if demand_model is not None:

            predictions = np.asarray(

                demand_model.get(
                    "full_predictions",
                    []
                )

            )

            if len(predictions) == len(
                intelligence_df
            ):

                intelligence_df[
                    "Predicted Demand"
                ] = predictions

        # =================================================
        # COST PREDICTIONS
        # =================================================

        cost_model = pipeline_output.get(
            "cost_model"
        )

        if cost_model is not None:

            predictions = np.asarray(

                cost_model.get(
                    "full_predictions",
                    []
                )

            )

            if len(predictions) == len(
                intelligence_df
            ):

                intelligence_df[
                    "Predicted Cost"
                ] = predictions

        # =================================================
        # SAFETY PREDICTIONS
        # =================================================

        safety_model = pipeline_output.get(
            "safety_model"
        )

        if safety_model is not None:

            predictions = np.asarray(

                safety_model.get(
                    "full_predictions",
                    []
                )

            )

            if len(predictions) == len(
                intelligence_df
            ):

                intelligence_df[
                    "Predicted Safety"
                ] = predictions

        # =================================================
        # SAFETY SCORE
        # =================================================

        if "Predicted Safety" in intelligence_df.columns:
            
            print("\nSafety Predictions Distribution:")
            
            print(intelligence_df["Predicted Safety"].value_counts(dropna=False))
            
            safety_series = (intelligence_df["Predicted Safety"]
                             .astype(str)
                             .str.lower()
                             .str.strip())
            
            safety_map = {
                "low risk": 100,
                "low": 100,
                "safe": 100,
                "pass": 100,
                "good": 100,

                "medium risk": 70,
                "medium": 70,
                "moderate": 70,

                "high risk": 30,
                "high": 30,
                "unsafe": 0,

                "0": 100,
                "1": 70,
                "2": 30}
            
            intelligence_df["Safety Score"] = (safety_series
                                               .map(safety_map).fillna(50))
        
        else:
            
            intelligence_df["Safety Score"] = 50

        # =================================================
        # REQUIRED COLUMNS
        # =================================================

        required_columns = [

            entity_column,

            supplier_column
        ]

        if "Predicted Demand" in intelligence_df.columns:

            required_columns.append(
                "Predicted Demand"
            )

        if "Predicted Cost" in intelligence_df.columns:

            required_columns.append(
                "Predicted Cost"
            )

        required_columns.append(
            "Safety Score"
        )

        if (inhouse_cost_column
            and 
            inhouse_cost_column in intelligence_df.columns):
            
            required_columns.append(inhouse_cost_column)
            
        if (inhouse_capacity_column
            and
            inhouse_capacity_column in intelligence_df.columns):
            
            required_columns.append(inhouse_capacity_column)
            
        if (inhouse_safety_column
            and
            inhouse_safety_column in intelligence_df.columns):
            
            required_columns.append(inhouse_safety_column)

        supplier_df = (

            intelligence_df[
                required_columns
            ]

            .copy()
        )

        # =================================================
        # SUPPLIER AGGREGATION
        # =================================================

        aggregation = {}

        if "Predicted Demand" in supplier_df.columns:

            aggregation[
                "Predicted Demand"
            ] = "mean"

        if "Predicted Cost" in supplier_df.columns:

            aggregation[
                "Predicted Cost"
            ] = "mean"

        if (
            inhouse_cost_column
            and
            inhouse_cost_column in supplier_df.columns):
            
            aggregation[inhouse_cost_column] = "mean"
            
        if (
            inhouse_capacity_column
            and
            inhouse_capacity_column in supplier_df.columns):
            
            aggregation[inhouse_capacity_column] = "mean"

        if (
            inhouse_safety_column
            and
            inhouse_safety_column in supplier_df.columns):
            
            aggregation[inhouse_safety_column] = "mean"

        aggregation[
            "Safety Score"
        ] = "mean"

        supplier_df = (

            supplier_df

            .groupby(

                [
                    entity_column,
                    supplier_column
                ],

                as_index=False

            )

            .agg(
                aggregation
            )
        )

        # =================================================
        # PART LEVEL ANALYSIS
        # =================================================

        results = []

        for part_name, part_df in (

            supplier_df.groupby(
                entity_column
            )

        ):

            part_df = part_df.copy()

            # =============================================
            # DEMAND RANK
            # =============================================

            if "Predicted Demand" in part_df.columns:

                demand_min = (

                    part_df[
                        "Predicted Demand"
                    ].min()
                )

                demand_max = (

                    part_df[
                        "Predicted Demand"
                    ].max()
                )

                if demand_max > demand_min:

                    part_df[
                        "Demand Rank"
                    ] = (

                        (

                            part_df[
                                "Predicted Demand"
                            ]

                            - demand_min

                        )

                        /

                        (

                            demand_max
                            - demand_min
                        )

                    ) * 100

                else:

                    part_df[
                        "Demand Rank"
                    ] = 100

            else:

                part_df[
                    "Demand Rank"
                ] = 50

            # =============================================
            # COST RANK
            # =============================================

            if "Predicted Cost" in part_df.columns:

                cost_min = (

                    part_df[
                        "Predicted Cost"
                    ].min()
                )

                cost_max = (

                    part_df[
                        "Predicted Cost"
                    ].max()
                )

                if cost_max > cost_min:

                    part_df[
                        "Cost Rank"
                    ] = (

                        1 -

                        (

                            (

                                part_df[
                                    "Predicted Cost"
                                ]

                                - cost_min

                            )

                            /

                            (

                                cost_max
                                - cost_min
                            )

                        )

                    ) * 100

                else:

                    part_df[
                        "Cost Rank"
                    ] = 100

            else:

                part_df[
                    "Cost Rank"
                ] = 50

            # =============================================
            # PROCUREMENT SCORE
            # =============================================

            part_df[
                "Procurement Score"
            ] = (

                part_df[
                    "Demand Rank"
                ] * 0.40

                +

                part_df[
                    "Cost Rank"
                ] * 0.35

                +

                part_df[
                    "Safety Score"
                ] * 0.25
            )

            # =============================================
            # BEST DEMAND SUPPLIER
            # =============================================

            demand_supplier = None
            demand_value = None

            if "Predicted Demand" in part_df.columns:

                demand_row = (

                    part_df.loc[
                        part_df[
                            "Predicted Demand"
                        ].idxmax()
                    ]
                )

                demand_supplier = demand_row[
                    supplier_column
                ]

                demand_value = round(

                    float(

                        demand_row[
                            "Predicted Demand"
                        ]

                    ),

                    2
                )

            # =============================================
            # BEST COST SUPPLIER
            # =============================================

            cost_supplier = None
            cost_value = None

            if "Predicted Cost" in part_df.columns:

                cost_row = (

                    part_df.loc[
                        part_df[
                            "Predicted Cost"
                        ].idxmin()
                    ]
                )

                cost_supplier = cost_row[
                    supplier_column
                ]

                cost_value = round(

                    float(

                        cost_row[
                            "Predicted Cost"
                        ]

                    ),

                    2
                )

            # =============================================
            # BEST SAFETY SUPPLIER
            # =============================================

            safety_row = (

                part_df.loc[
                    part_df[
                        "Safety Score"
                    ].idxmax()
                ]
            )

            # =============================================
            # FINAL RECOMMENDED SUPPLIER
            # =============================================

            recommended_row = (

                part_df.loc[
                    part_df[
                        "Procurement Score"
                    ].idxmax()
                ]
            )

            # =============================================
            # PROCUREMENT SITUATION
            # =============================================

            demand_score = float(

                recommended_row.get(
                    "Demand Rank",
                    50
                )
            )

            cost_score = float(

                recommended_row.get(
                    "Cost Rank",
                    50
                )
            )

            safety_score = float(

                recommended_row.get(
                    "Safety Score",
                    50
                )
            )

            if safety_score <= 50:

                procurement_situation = (
                    "Safety Risk Procurement"
                )

            elif demand_score >= 80 and cost_score <= 50:

                procurement_situation = (
                    "High Demand / High Cost Risk"
                )

            elif demand_score >= 80 and safety_score >= 80:

                procurement_situation = (
                    "Strategic Procurement Opportunity"
                )

            elif cost_score >= 80 and safety_score >= 80:

                procurement_situation = (
                    "Cost Efficient Procurement"
                )

            else:

                procurement_situation = (
                    "Balanced Procurement"
                )

            # =============================================
            # INHOUSE PROCUREMENT CHECK
            # =============================================

            procurement_mode = (
                "External Supplier"
            )

            if (
                inhouse_cost_column
                and
                inhouse_capacity_column
                and
                inhouse_safety_column
                and
                inhouse_cost_column in part_df.columns
                and
                inhouse_capacity_column in part_df.columns
                and
                inhouse_safety_column in part_df.columns):

                supplier_procurement_score = float(

                    recommended_row[
                        "Procurement Score"
                    ]
                )

                supplier_cost = float(

                    recommended_row.get(
                        "Predicted Cost",
                        0
                    )
                )

                supplier_demand = float(

                    recommended_row.get(
                        "Predicted Demand",
                        0
                    )
                )

                inhouse_cost = float(

                    part_df[
                        inhouse_cost_column
                    ].mean()
                )

                inhouse_capacity = float(

                    part_df[
                        inhouse_capacity_column
                    ].mean()
                )

                inhouse_safety = float(

                    part_df[
                        inhouse_safety_column
                    ].mean()
                )

                if supplier_cost > 0 and inhouse_cost > 0:

                    inhouse_cost_score = min(

                        100,

                        (
                            supplier_cost
                            /
                            inhouse_cost
                        ) * 100
                    )

                else:

                    inhouse_cost_score = 50

                if supplier_demand > 0 and inhouse_capacity > 0:

                    inhouse_capacity_score = min(

                        100,

                        (
                            inhouse_capacity
                            /
                            supplier_demand
                        ) * 100
                    )

                else:

                    inhouse_capacity_score = 50

                inhouse_procurement_score = (

                    0.40 * inhouse_capacity_score

                    +

                    0.35 * inhouse_cost_score

                    +

                    0.25 * inhouse_safety

                )

                # ==========================================
                # PROCUREMENT DECISION INDEX
                # ==========================================
                
                score_gap = (inhouse_procurement_score-supplier_procurement_score)
            
                # Strong In-House Advantage
             
                if (
                    score_gap >= 10
                    and
                    inhouse_capacity >= 0.90 * supplier_demand):
                    
                    procurement_mode = "In-House"
                    
                 # Strong Supplier Advantage
                
                elif (score_gap <= -10):
                    
                    procurement_mode = "External Supplier"

                 # Close Competition
                
                else:
                    
                    procurement_mode = "Hybrid"
                    
            results.append({

                "Part Name":
                part_name,

                "Best Demand Supplier":
                demand_supplier,

                "Best Demand":
                demand_value,

                "Best Cost Supplier":
                cost_supplier,

                "Best Cost":
                cost_value,

                "Best Safety Supplier":
                safety_row[
                    supplier_column
                ],

                "Recommended Demand":
                round(

                    float(

                        recommended_row.get(
                            "Predicted Demand",
                            50
                        )

                    ),

                    2
                ),

                "Recommended Cost":
                round(

                    float(

                        recommended_row.get(
                            "Predicted Cost",
                            50
                        )

                    ),

                    2
                ),

                "Recommended Safety":
                round(

                    float(

                        recommended_row.get(
                            "Safety Score",
                            50
                        )

                    ),

                    2
                ),

                "Demand Score":
                round(

                    float(

                        recommended_row.get(
                            "Demand Rank",
                            50
                        )

                    ),

                    2
                ),

                "Cost Score":
                round(

                    float(

                        recommended_row.get(
                            "Cost Rank",
                            50
                        )

                    ),

                    2
                ),

                "Safety Score":
                round(

                    float(

                        recommended_row.get(
                            "Safety Score",
                            50
                        )

                    ),

                    2
                ),

                "Procurement Situation":
                procurement_situation,

                "Procurement Mode":
                procurement_mode,

                "Recommended Supplier":
                recommended_row[
                    supplier_column
                ],

                "Procurement Score":
                round(

                    float(

                        recommended_row[
                            "Procurement Score"
                        ]

                    ),

                    2
                )
            })

        supplier_intelligence_df = pd.DataFrame(
            results
        )

        required_columns = [
            "Part Name",

            "Best Demand Supplier",
            "Best Demand",

            "Best Cost Supplier",
            "Best Cost",

            "Best Safety Supplier",

            "Recommended Demand",
            "Recommended Cost",
            "Recommended Safety",

            "Demand Score",
            "Cost Score",
            "Safety Score",

            "Procurement Situation",
            "Procurement Mode",
            "Recommended Supplier",
            
            "Procurement Score"]
        
        for col in required_columns:
            
            if col not in supplier_intelligence_df.columns:
                
                supplier_intelligence_df[col] = np.nan

        print("\n================================")
        print("SUPPLIER INTELLIGENCE GENERATED")
        print("================================")
        print(
            f"Parts: {len(supplier_intelligence_df)}"
        )
        print("================================\n")

        supplier_intelligence_df = supplier_intelligence_df[required_columns]

        print("EXITING build_supplier_intelligence")

        return supplier_intelligence_df

    except Exception as e:
        print("\n========== SUPPLIER FAILURE ==========")
        print(e)
        traceback.print_exc()
        print("======================================\n")
        return pd.DataFrame()

# =========================================================
# MAIN ENTERPRISE PIPELINE
# =========================================================

def run_enterprise_pipeline(

    processed_df,
    raw_business_df,

    targets,
    task_types,

    feature_columns,

    entity_column=None,
    supplier_column=None

):

    # =====================================================
    # OUTPUT
    # =====================================================

    pipeline_output = {

        "errors": {}
    }

    # =====================================================
    # VALIDATION
    # =====================================================

    if processed_df is None:

        raise ValueError(
            "Processed dataframe is None"
        )

    if raw_business_df is None:

        raise ValueError(
            "Raw dataframe is None"
        )

    if processed_df.empty:

        raise ValueError(
            "Processed dataframe empty"
        )

    if raw_business_df.empty:

        raise ValueError(
            "Raw dataframe empty"
        )

    if len(processed_df) != len(raw_business_df):

        raise ValueError(

            f"Dataset Row Mismatch -> "

            f"Processed={len(processed_df)} "

            f"Raw={len(raw_business_df)}"
        )

    # =====================================================
    # TARGETS
    # =====================================================

    demand_target = targets.get(
        "demand_target"
    )

    cost_target = targets.get(
        "cost_target"
    )

    safety_target = targets.get(
        "safety_target"
    )

    target_columns = [

        col

        for col in [

            demand_target,
            cost_target,
            safety_target

        ]

        if (

            col is not None

            and

            col in processed_df.columns
        )
    ]

    # =====================================================
    # FEATURES
    # =====================================================

    X_columns = [

        col

        for col in feature_columns

        if (

            col in processed_df.columns

            and

            col not in target_columns
        )
    ]

    if len(X_columns) == 0:

        raise ValueError(
            "No feature columns available"
        )

    # =====================================================
    # FEATURE MATRIX
    # =====================================================

    X = processed_df[
        X_columns
    ].copy()

    X = X.replace(

        [np.inf, -np.inf],

        np.nan
    )

    X = X.fillna(0)

    X = X.reset_index(
        drop=True
    )

    # =====================================================
    # PIPELINE SUMMARY
    # =====================================================

    print("\n================================")
    print("PIPELINE DATASET SUMMARY")
    print("================================")

    print(
        "Rows:",
        len(X)
    )

    print(
        "Feature Count:",
        len(X_columns)
    )

    print(
        "Entity Column:",
        entity_column
    )

    print(
        "Supplier Column:",
        supplier_column
    )

    print(
        "Targets:",
        targets
    )

    print(
        "Task Types:",
        task_types
    )

    print("================================\n")

    # =====================================================
    # MODEL MAP
    # =====================================================

    model_targets = {

        "demand_model":
        demand_target,

        "cost_model":
        cost_target,

        "safety_model":
        safety_target
    }

    # =====================================================
    # TRAIN MODELS
    # =====================================================

    for model_key, target_column in model_targets.items():

        try:

            if target_column is None:

                continue

            if target_column not in raw_business_df.columns:

                pipeline_output[
                    "errors"
                ][model_key] = (

                    "Target Missing"
                )

                continue

            y = raw_business_df[
                target_column
            ].copy()

            y = y.reset_index(
                drop=True
            )

            # =============================================
            # VALIDATION
            # =============================================

            if len(X) != len(y):

                raise ValueError(

                    f"Length Mismatch -> "

                    f"X={len(X)} "

                    f"Y={len(y)}"
                )

            if y.isnull().all():

                pipeline_output[
                    "errors"
                ][model_key] = (

                    "Empty Target"
                )

                continue

            task_key = model_key.replace(

                "_model",

                "_target"
            )

            task_type = task_types.get(
                task_key
            )

            if task_type not in [

                "classification",
                "regression"
            ]:

                pipeline_output[
                    "errors"
                ][model_key] = (

                    f"Invalid Task Type ({task_type})"
                )

                continue

            print("\n================================")

            print(
                "MODEL:",
                model_key
            )

            print(
                "TARGET:",
                target_column
            )

            print(
                "TASK:",
                task_type
            )

            print(
                "ROWS:",
                len(y)
            )

            print(
                "UNIQUE:",
                y.nunique()
            )

            print("================================")

            # =============================================
            # CLASSIFICATION
            # =============================================

            if task_type == "classification":

                if y.nunique() < 2:

                    pipeline_output[
                        "errors"
                    ][model_key] = (

                        "Single Class Target"
                    )

                    continue

                model_result = (

                    train_classification_pipeline(

                        X,

                        y,

                        X_columns
                    )
                )

            # =============================================
            # REGRESSION
            # =============================================

            else:

                if y.nunique() <= 1:

                    pipeline_output[
                        "errors"
                    ][model_key] = (

                        "Constant Target"
                    )

                    continue

                model_result = (

                    train_regression_pipeline(

                        X,

                        y,

                        X_columns
                    )
                )

            if model_result is None:

                pipeline_output[
                    "errors"
                ][model_key] = (

                    "Training Failed"
                )

                continue

            pipeline_output[
                model_key
            ] = model_result

            print(
                f"{model_key} SUCCESS"
            )

        except Exception as e:

            print(
                f"{model_key} FAILED -> {e}"
            )

            pipeline_output[
                "errors"
            ][model_key] = str(e)

    # =====================================================
    # PROCUREMENT METADATA
    # =====================================================

    pipeline_output[
        "entity_column"
    ] = entity_column

    pipeline_output[
        "supplier_column"
    ] = supplier_column

    # =====================================================
    # SUPPLIER INTELLIGENCE DATASET
    # =====================================================

    supplier_intelligence_df = build_supplier_intelligence(
    raw_business_df=raw_business_df,
    pipeline_output=pipeline_output,
    entity_column=entity_column,
    supplier_column=supplier_column,
    targets=targets)

    print(
    "Supplier Intelligence Shape:",
    supplier_intelligence_df.shape)
    
    print("Supplier Intelligence Empty:",supplier_intelligence_df.empty)

    pipeline_output[
        "supplier_intelligence_df"
    ] = supplier_intelligence_df

    print(
    "Stored Supplier DF Shape:",
    pipeline_output[
        "supplier_intelligence_df"].shape)

    # =====================================================
    # GENERATED MODELS
    # =====================================================

    generated_models = [

        key

        for key

        in pipeline_output.keys()

        if key.endswith(
            "_model"
        )
    ]

    # =====================================================
    # DIAGNOSTICS
    # =====================================================

    pipeline_output[
        "diagnostics"
    ] = {

        "row_count":
        len(processed_df),

        "feature_count":
        len(X_columns),

        "targets":
        targets,

        "task_types":
        task_types,

        "entity_column":
        entity_column,

        "supplier_column":
        supplier_column,

        "generated_models":
        generated_models,

        "errors":
        pipeline_output[
            "errors"
        ]
    }

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("\n================================")
    print("PIPELINE SUMMARY")
    print("================================")

    print(
        "Entity:",
        entity_column
    )

    print(
        "Supplier:",
        supplier_column
    )

    print(
        "Generated Models:",
        generated_models
    )

    print(
        "Errors:",
        pipeline_output[
            "errors"
        ]
    )

    print("================================\n")

    return pipeline_output