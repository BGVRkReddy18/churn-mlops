import numpy as np
import pytest

from churn.features import MODEL_TYPES, build_pipeline, split_xy
from churn.train import best_f1_threshold


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_every_model_type_trains_and_predicts(clean_df, model_type):
    X, y = split_xy(clean_df)
    proba = build_pipeline(model_type, {}).fit(X, y).predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_ignores_unseen_category(clean_df):
    X, y = split_xy(clean_df)
    pipe = build_pipeline("logreg", {}).fit(X, y)
    X_new = X.head(1).copy()
    X_new["PaymentMethod"] = "Crypto"
    pipe.predict_proba(X_new)  # must not raise


def test_unknown_model_type_raises():
    with pytest.raises(ValueError):
        build_pipeline("xgboost-ultra", {})


def test_best_f1_threshold_separates_perfect_classes():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert 0.3 < best_f1_threshold(y, proba) <= 0.8
