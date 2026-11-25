import json

import joblib
import numpy as np
import pandas as pd
import pytest

from churn.features import build_pipeline
from churn.schema import CATEGORY_VALUES, TARGET


def make_raw_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic data in the *raw* CSV format (SeniorCitizen 0/1, TotalCharges as text)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.choice(vals, n) for col, vals in CATEGORY_VALUES.items()})
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 72, n)
    df["MonthlyCharges"] = rng.uniform(18, 120, n).round(2)
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2).astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    df["customerID"] = [f"C{i}" for i in range(n)]
    df[TARGET] = rng.choice(["No", "Yes"], n, p=[0.7, 0.3])
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_df()


@pytest.fixture
def clean_df(raw_df) -> pd.DataFrame:
    from churn.data.prepare import clean

    return clean(raw_df)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """A small model trained on synthetic data so API tests don't depend on the DVC pipeline."""
    from churn.data.prepare import clean
    from churn.features import split_xy

    X, y = split_xy(clean(make_raw_df(400)))
    pipe = build_pipeline("logreg", {"C": 1.0}).fit(X, y)
    d = tmp_path_factory.mktemp("model")
    joblib.dump(pipe, d / "model.joblib")
    (d / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "churn-classifier",
                "model_version": "test",
                "model_type": "logreg",
                "threshold": 0.5,
                "trained_at": "2026-01-01T00:00:00+00:00",
                "metrics": {"test_roc_auc": 0.5},
            }
        )
    )
    return d
