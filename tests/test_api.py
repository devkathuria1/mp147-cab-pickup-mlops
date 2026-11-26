import io
import json
import logging
import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.main import app, FEATURES

VALID_BODY = {
    "drivers_nearby": 3,
    "distance_to_driver_km": 2.5,
    "hour": 18,
    "rain_mm": 5,
    "surge": 1.2,
    "road_closures": 0,
    "car_type": "sedan",
    "zone": "Airport",
}


def build_tiny_model(path):
    """Fit a small Pipeline on 20 FAKE rows (no CSV) and save it to `path`."""
    cars = ["mini", "sedan", "auto"]
    zones = ["Airport", "Old City", "Tech Park"]
    rows = []
    for i in range(20):
        rows.append([
            i % 5, 1.0 + i * 0.3, i % 24, float(i % 4), 1.0 + (i % 3) * 0.1,
            i % 2, cars[i % 3], zones[i % 3],
        ])
    X = pd.DataFrame(rows, columns=FEATURES)
    y = [5.0 + i * 0.5 for i in range(20)]

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["car_type", "zone"])],
        remainder="passthrough",
    )
    model = Pipeline([("pre", pre), ("reg", RandomForestRegressor(n_estimators=5, random_state=42))])
    model.fit(X, y)
    joblib.dump(model, path)


@pytest.fixture
def client(tmp_path, monkeypatch):
    model_file = tmp_path / "tiny_model.joblib"
    build_tiny_model(model_file)
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    with TestClient(app) as c:   # "with" runs the startup (lifespan) code
        yield c


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_returns_number(client):
    response = client.post("/predict", json=VALID_BODY)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_predict_rain_above_85_returns_422(client):
    body = {**VALID_BODY, "rain_mm": 90}
    response = client.post("/predict", json=body)
    assert response.status_code == 422


def test_predict_negative_distance_returns_422(client):
    body = {**VALID_BODY, "distance_to_driver_km": -1}
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    
def test_prediction_is_logged_as_json(client):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    from app.main import JsonFormatter, logger
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    try:
        client.post("/predict", json=VALID_BODY)
    finally:
        logger.removeHandler(handler)

    line = stream.getvalue().strip().splitlines()[-1]
    entry = json.loads(line)              # must be valid JSON
    assert entry["message"] == "prediction"
    assert entry["zone"] == "Airport"
    assert "prediction" in entry