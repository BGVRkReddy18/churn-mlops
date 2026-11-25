"""Stage 3 - hyperparameter search with Optuna, tracked in MLflow."""

import json
import logging

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from churn.config import (
    MODEL_DIR,
    MODEL_PATH,
    REPORTS_DIR,
    TRAIN_INFO_PATH,
    TRAIN_PATH,
    load_params,
    mlflow_tracking_uri,
)
from churn.features import build_pipeline, split_xy, suggest_hyperparams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def best_f1_threshold(y_true, proba) -> float:
    """Pick the probability cut-off that maximises F1 (default 0.5 is rarely optimal
    for an imbalanced target like churn)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def main() -> None:
    params = load_params()
    tp = params["train"]
    model_type, seed = tp["model"], tp["random_state"]

    X, y = split_xy(pd.read_csv(TRAIN_PATH))
    cv = StratifiedKFold(n_splits=tp["cv_folds"], shuffle=True, random_state=seed)

    mlflow.set_tracking_uri(mlflow_tracking_uri())
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"train-{model_type}") as run:
        mlflow.set_tag("stage", "train")

        def objective(trial: optuna.Trial) -> float:
            hp = suggest_hyperparams(trial, model_type)
            pipe = build_pipeline(model_type, hp, seed)
            score = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
            # each trial becomes a nested MLflow run -> compare them in the UI
            with mlflow.start_run(run_name=f"trial-{trial.number:02d}", nested=True):
                mlflow.log_params(hp)
                mlflow.log_metric("cv_roc_auc", score)
            return score

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=tp["n_trials"])
        best_hp = study.best_params
        log.info("Best CV ROC-AUC %.4f with %s", study.best_value, best_hp)

        pipe = build_pipeline(model_type, best_hp, seed)
        # out-of-fold predictions -> choose decision threshold without touching the test set
        oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        threshold = best_f1_threshold(y, oof)
        pipe.fit(X, y)

        mlflow.log_params({"model_type": model_type, "n_trials": tp["n_trials"], **best_hp})
        mlflow.log_metrics({"best_cv_roc_auc": study.best_value, "threshold": threshold})

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, MODEL_PATH)
        TRAIN_INFO_PATH.write_text(
            json.dumps(
                {
                    "run_id": run.info.run_id,
                    "model_type": model_type,
                    "best_params": best_hp,
                    "cv_roc_auc": study.best_value,
                    "threshold": threshold,
                },
                indent=2,
            )
        )
    log.info("Model saved to %s (threshold=%.3f)", MODEL_PATH, threshold)


if __name__ == "__main__":
    main()
