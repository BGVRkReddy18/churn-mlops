"""Data-drift check: compare live API inputs against the training data using the
Population Stability Index (PSI).  PSI < 0.1 stable | 0.1-0.2 moderate | > 0.2 significant.

Usage:  python -m churn.monitoring.drift [--log logs/predictions.jsonl] [--fail-on-drift]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import REPORTS_DIR, ROOT, TRAIN_PATH
from churn.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

EPS = 1e-6
DRIFT_THRESHOLD = 0.2


def _psi(ref_pct: np.ndarray, cur_pct: np.ndarray) -> float:
    ref_pct, cur_pct = np.clip(ref_pct, EPS, None), np.clip(cur_pct, EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, edges)[0] / len(ref)
    cur_pct = np.histogram(cur, edges)[0] / len(cur)
    return _psi(ref_pct, cur_pct)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref) | set(cur))
    ref_pct = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    cur_pct = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(ref_pct, cur_pct)


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    features = {}
    for col in NUMERIC_FEATURES:
        features[col] = psi_numeric(reference[col], current[col])
    for col in CATEGORICAL_FEATURES:
        features[col] = psi_categorical(reference[col], current[col])
    drifted = sorted(c for c, v in features.items() if v > DRIFT_THRESHOLD)
    return {
        "n_reference": len(reference),
        "n_current": len(current),
        "threshold": DRIFT_THRESHOLD,
        "psi": {k: round(v, 4) for k, v in features.items()},
        "drifted_features": drifted,
        "drift_detected": bool(drifted),
    }


def load_prediction_log(path: Path) -> pd.DataFrame:
    rows = [json.loads(line)["features"] for line in path.read_text().splitlines() if line]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=ROOT / "logs" / "predictions.jsonl")
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()

    if not args.log.exists():
        sys.exit(f"No prediction log at {args.log} - send some traffic to the API first.")
    current = load_prediction_log(args.log)
    if len(current) < args.min_samples:
        sys.exit(f"Only {len(current)} predictions logged; need >= {args.min_samples}.")

    report = drift_report(pd.read_csv(TRAIN_PATH), current)
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / "drift_report.json"
    out.write_text(json.dumps(report, indent=2))

    log.info("%-18s %8s", "feature", "PSI")
    for col, v in sorted(report["psi"].items(), key=lambda kv: -kv[1]):
        flag = "  <-- DRIFT" if v > DRIFT_THRESHOLD else ""
        log.info("%-18s %8.4f%s", col, v, flag)
    log.info("\nDrift detected: %s (report: %s)", report["drift_detected"], out)

    if args.fail_on_drift and report["drift_detected"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
