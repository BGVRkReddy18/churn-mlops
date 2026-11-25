import pytest
from fastapi.testclient import TestClient

from churn.api.schemas import Customer

EXAMPLE = Customer.model_config["json_schema_extra"]["example"]


@pytest.fixture
def client(model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "predictions.jsonl"))
    from churn.api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_version"] == "test"
    assert body["threshold"] == 0.5


def test_predict(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["churn"] == (body["churn_probability"] >= body["threshold"])


def test_predict_batch(client):
    r = client.post("/predict/batch", json={"customers": [EXAMPLE, EXAMPLE]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


@pytest.mark.parametrize(
    "field,value", [("Contract", "Lifetime"), ("tenure", -1), ("MonthlyCharges", "abc")]
)
def test_predict_rejects_invalid_input(client, field, value):
    assert client.post("/predict", json={**EXAMPLE, field: value}).status_code == 422


def test_predict_rejects_empty_batch(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_predictions_are_logged(client, tmp_path):
    client.post("/predict", json=EXAMPLE)
    assert (tmp_path / "predictions.jsonl").read_text().count("\n") == 1


def test_metrics_endpoint(client):
    client.post("/predict", json=EXAMPLE)
    text = client.get("/metrics").text
    assert "churn_predictions_total" in text
    assert "http_requests_total" in text


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Churn MLOps" in r.text and "/predict" in r.text
