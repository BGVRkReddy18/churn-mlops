"""Feature engineering + model factory. Everything lives in ONE sklearn Pipeline,
so the exact same preprocessing runs in training and in the API (no train/serve skew)."""

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.schema import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

MODEL_TYPES = ("logreg", "random_forest", "hist_gb")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], df[TARGET]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_model(model_type: str, hyperparams: dict[str, Any], random_state: int = 42):
    if model_type == "logreg":
        return LogisticRegression(max_iter=2000, **hyperparams)
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=random_state, n_jobs=-1, **hyperparams)
    if model_type == "hist_gb":
        return HistGradientBoostingClassifier(random_state=random_state, **hyperparams)
    raise ValueError(f"Unknown model type {model_type!r}; expected one of {MODEL_TYPES}")


def build_pipeline(model_type: str, hyperparams: dict[str, Any], random_state: int = 42):
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", build_model(model_type, hyperparams, random_state)),
        ]
    )


def suggest_hyperparams(trial, model_type: str) -> dict[str, Any]:
    """Optuna search space for each model type."""
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    if model_type == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True), "class_weight": class_weight}
    if model_type == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "class_weight": class_weight,
        }
    if model_type == "hist_gb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
            "class_weight": class_weight,
        }
    raise ValueError(f"Unknown model type {model_type!r}")
