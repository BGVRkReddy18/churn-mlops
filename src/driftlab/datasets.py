"""Datasets used in the study.

* telco        - IBM Telco churn (same data as the production system). Drift is injected
                 synthetically, so the drift onset is known exactly (ground truth).
* electricity  - Elec2 (Harries 1999), NSW electricity market 1996-1998. A standard benchmark
                 with *real*, naturally occurring drift (no ground truth onset).
"""

import io
from dataclasses import dataclass, field

import httpx
import numpy as np
import pandas as pd

from churn.config import RAW_DATA_PATH, ROOT
from churn.data.prepare import clean
from churn.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

ELECTRICITY_URL = "https://api.openml.org/data/v1/download/2419"
ELECTRICITY_PATH = ROOT / "data" / "raw" / "electricity.arff"
ELECTRICITY_COLUMNS = [
    "date", "day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer", "class",
]  # fmt: skip


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    numeric: list[str]
    categorical: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical

    def subset(self, idx) -> "Dataset":
        return Dataset(
            self.name,
            self.X.iloc[idx].reset_index(drop=True),
            self.y[idx],
            self.numeric,
            self.categorical,
            self.meta,
        )

    def __len__(self) -> int:
        return len(self.y)


def load_telco() -> Dataset:
    df = clean(pd.read_csv(RAW_DATA_PATH))
    return Dataset(
        "telco",
        df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].reset_index(drop=True),
        df[TARGET].to_numpy(),
        list(NUMERIC_FEATURES),
        list(CATEGORICAL_FEATURES),
    )


def download_electricity() -> None:
    if ELECTRICITY_PATH.exists():
        return
    ELECTRICITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    r = httpx.get(ELECTRICITY_URL, timeout=120, follow_redirects=True)
    r.raise_for_status()
    ELECTRICITY_PATH.write_bytes(r.content)


def load_electricity() -> Dataset:
    download_electricity()
    text = ELECTRICITY_PATH.read_text()
    body = text[text.lower().index("@data") + len("@data") :]
    df = pd.read_csv(io.StringIO(body), header=None, names=ELECTRICITY_COLUMNS)
    df["day"] = df["day"].astype(str)
    numeric = ["period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
    return Dataset(
        "electricity",
        df[numeric + ["day"]],
        (df["class"].str.strip() == "UP").astype(int).to_numpy(),
        numeric,
        ["day"],
        meta={"date": df["date"].to_numpy()},
    )


def make_encoder(ds: Dataset):
    """Numeric -> standardised, categorical -> one-hot. Used by the model and by
    multivariate detectors (MMD, classifier two-sample test)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return ColumnTransformer(
        [
            ("num", StandardScaler(), ds.numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ds.categorical),
        ]
    )


def make_model(ds: Dataset, random_state: int = 0):
    """The monitored production model (gradient boosting, as in the churn system)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("pre", make_encoder(ds)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=150,
                    learning_rate=0.08,
                    max_depth=4,
                    min_samples_leaf=30,
                    random_state=random_state,
                ),
            ),
        ]
    )
