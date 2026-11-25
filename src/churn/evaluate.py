"""Stage 4 - evaluate on the held-out test set, apply the quality gate,
register the model in the MLflow Model Registry and promote it if it beats the champion."""

import json
import logging
import sys
from datetime import UTC, datetime

import joblib
import mlflow
import pandas as pd
from matplotlib.figure import Figure
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn.config import (
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PLOTS_DIR,
    TEST_PATH,
    TRAIN_INFO_PATH,
    load_params,
    mlflow_tracking_uri,
)
from churn.features import split_xy
from churn.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
CHAMPION = "champion"


def compute_metrics(y_true, proba, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "test_roc_auc": roc_auc_score(y_true, proba),
        "test_pr_auc": average_precision_score(y_true, proba),
        "test_f1": f1_score(y_true, pred),
        "test_precision": precision_score(y_true, pred),
        "test_recall": recall_score(y_true, pred),
        "test_accuracy": accuracy_score(y_true, pred),
    }


def save_plots(y_true, proba, threshold: float) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(5, 4))
    ax = fig.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_true, (proba >= threshold).astype(int), display_labels=["Stay", "Churn"], ax=ax
    )
    ax.set_title(f"Confusion matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=120)

    fig = Figure(figsize=(5, 4))
    ax = fig.subplots()
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax)
    ax.set_title("ROC curve (test set)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curve.png", dpi=120)


def trusted_types(model) -> list[str]:
    """MLflow serialises sklearn models with skops, which asks us to list trusted types.
    We trained this model ourselves in this run, so trusting its own types is safe."""
    try:
        import skops.io as sio
    except ImportError:
        return []
    return sio.get_untrusted_types(data=sio.dumps(model))


def champion_score(client: MlflowClient, name: str) -> float | None:
    try:
        mv = client.get_model_version_by_alias(name, CHAMPION)
    except MlflowException:
        return None
    return client.get_run(mv.run_id).data.metrics.get("test_roc_auc")


def main() -> None:
    params = load_params()
    ep = params["evaluate"]
    train_info = json.loads(TRAIN_INFO_PATH.read_text())
    threshold = train_info["threshold"]

    model = joblib.load(MODEL_PATH)
    X, y = split_xy(pd.read_csv(TEST_PATH))
    proba = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(y, proba, threshold)
    save_plots(y, proba, threshold)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    for k, v in metrics.items():
        log.info("%-15s %.4f", k, v)

    mlflow.set_tracking_uri(mlflow_tracking_uri())
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    client = MlflowClient()
    name = ep["registered_model_name"]

    with mlflow.start_run(run_id=train_info["run_id"]):
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(PLOTS_DIR), artifact_path="plots")

        # ---- quality gate ----
        if metrics["test_roc_auc"] < ep["min_roc_auc"]:
            mlflow.set_tag("quality_gate", "failed")
            log.error(
                "Quality gate FAILED: test ROC-AUC %.4f < %.2f - model NOT registered",
                metrics["test_roc_auc"],
                ep["min_roc_auc"],
            )
            sys.exit(1)
        mlflow.set_tag("quality_gate", "passed")

        info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=name,
            input_example=X.head(3),
            skops_trusted_types=trusted_types(model),
        )
        version = info.registered_model_version

    # ---- promotion: only replace the champion if the new model is at least as good ----
    previous = champion_score(client, name)
    promoted = previous is None or metrics["test_roc_auc"] >= previous
    if promoted:
        client.set_registered_model_alias(name, CHAMPION, version)
        log.info("Promoted %s v%s to @%s (previous: %s)", name, version, CHAMPION, previous)
    else:
        log.warning(
            "v%s (%.4f) did not beat champion (%.4f)", version, metrics["test_roc_auc"], previous
        )

    METADATA_PATH.write_text(
        json.dumps(
            {
                "model_name": name,
                "model_version": str(version),
                "promoted_to_champion": promoted,
                "run_id": train_info["run_id"],
                "model_type": train_info["model_type"],
                "threshold": threshold,
                "metrics": metrics,
                "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
