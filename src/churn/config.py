"""Central place for paths and parameters used by the training pipeline."""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "params.yaml"

RAW_DATA_PATH = ROOT / "data" / "raw" / "churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

REPORTS_DIR = ROOT / "reports"
TRAIN_INFO_PATH = REPORTS_DIR / "train_info.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
PLOTS_DIR = REPORTS_DIR / "plots"


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def mlflow_tracking_uri() -> str:
    """Local SQLite store by default; point MLFLOW_TRACKING_URI at a server to share runs."""
    return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
