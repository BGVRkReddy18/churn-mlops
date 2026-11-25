"""Loads the trained pipeline + metadata and turns customers into predictions."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from churn.schema import FEATURES

log = logging.getLogger(__name__)


@dataclass
class ModelService:
    pipeline: object
    metadata: dict

    @classmethod
    def load(cls, model_dir: Path) -> "ModelService":
        pipeline = joblib.load(model_dir / "model.joblib")
        meta_path = model_dir / "metadata.json"
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        log.info("Loaded model %s v%s", metadata.get("model_name"), metadata.get("model_version"))
        return cls(pipeline=pipeline, metadata=metadata)

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("threshold", 0.5))

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    def predict_proba(self, records: list[dict]) -> list[float]:
        df = pd.DataFrame(records)[FEATURES]
        return self.pipeline.predict_proba(df)[:, 1].tolist()
