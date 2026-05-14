from sklearn.ensemble import (

    RandomForestRegressor,
    RandomForestClassifier,

    GradientBoostingRegressor,
    GradientBoostingClassifier,

    ExtraTreesRegressor,
    ExtraTreesClassifier
)

from xgboost import (

    XGBRegressor,
    XGBClassifier
)

# ==========================================
# MODEL SELECTOR
# ==========================================

def select_model(problem_type):

    # ======================================
    # REGRESSION MODELS
    # ======================================

    if problem_type == "regression":

        model = XGBRegressor(

            n_estimators=300,

            learning_rate=0.05,

            max_depth=8,

            subsample=0.8,

            colsample_bytree=0.8,

            random_state=42
        )

    # ======================================
    # CLASSIFICATION MODELS
    # ======================================

    else:

        model = XGBClassifier(

            n_estimators=300,

            learning_rate=0.05,

            max_depth=6,

            subsample=0.8,

            colsample_bytree=0.8,

            eval_metric='mlogloss',

            random_state=42
        )

    return model