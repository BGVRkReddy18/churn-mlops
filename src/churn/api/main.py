"""FastAPI prediction service with Prometheus metrics and prediction logging."""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from churn.api.model_service import ModelService
from churn.api.schemas import BatchRequest, BatchResponse, Customer, ModelInfo, Prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("churn.api")

PREDICTIONS = Counter("churn_predictions_total", "Predictions served", ["outcome"])
PROBABILITY = Histogram(
    "churn_probability", "Predicted churn probability", buckets=[i / 10 for i in range(1, 11)]
)
INFERENCE_SECONDS = Histogram("churn_inference_seconds", "Model inference time")
STATIC_DIR = Path(__file__).parent / "static"


def _model_dir() -> Path:
    return Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[3] / "models"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ModelService.load(_model_dir())
    log_path = os.getenv("PREDICTION_LOG_PATH")  # set to enable logging for drift monitoring
    app.state.log_path = Path(log_path) if log_path else None
    if app.state.log_path:
        app.state.log_path.parent.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Predicts the probability that a telecom customer will churn.",
    lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app, include_in_schema=False)


def _service(request: Request) -> ModelService:
    service = getattr(request.app.state, "model", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return service


def _predict(request: Request, customers: list[Customer]) -> list[Prediction]:
    service = _service(request)
    records = [c.model_dump() for c in customers]

    start = time.perf_counter()
    probas = service.predict_proba(records)
    INFERENCE_SECONDS.observe(time.perf_counter() - start)

    results = []
    for record, p in zip(records, probas, strict=True):
        churn = p >= service.threshold
        PREDICTIONS.labels(outcome="churn" if churn else "stay").inc()
        PROBABILITY.observe(p)
        results.append(
            Prediction(
                churn=churn,
                churn_probability=round(p, 4),
                threshold=round(service.threshold, 4),
                model_version=service.version,
            )
        )
        _log_prediction(request, record, p)
    return results


def _log_prediction(request: Request, features: dict, proba: float) -> None:
    path = request.app.state.log_path
    if path is None:
        return
    entry = {"ts": datetime.now(UTC).isoformat(), "features": features, "proba": proba}
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/", include_in_schema=False)
def home():
    """Project landing page with a live prediction form."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["ops"])
def health(request: Request):
    return {"status": "ok", "model_loaded": getattr(request.app.state, "model", None) is not None}


@app.get("/model-info", response_model=ModelInfo, tags=["ops"])
def model_info(request: Request):
    m = _service(request).metadata
    return ModelInfo(
        model_name=m.get("model_name", "unknown"),
        model_version=str(m.get("model_version", "unknown")),
        model_type=m.get("model_type", "unknown"),
        threshold=float(m.get("threshold", 0.5)),
        trained_at=m.get("trained_at"),
        metrics=m.get("metrics", {}),
    )


@app.post("/predict", response_model=Prediction, tags=["predict"])
def predict(customer: Customer, request: Request):
    return _predict(request, [customer])[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["predict"])
def predict_batch(body: BatchRequest, request: Request):
    return BatchResponse(predictions=_predict(request, body.customers))
