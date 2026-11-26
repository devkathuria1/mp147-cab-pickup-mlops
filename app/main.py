import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

FEATURES = [
    "drivers_nearby", "distance_to_driver_km", "hour", "rain_mm",
    "surge", "road_closures", "car_type", "zone",
]
ROOT = Path(__file__).resolve().parent.parent


# ---------- JSON logging ----------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))   # extra fields passed by us
        return json.dumps(entry)


logger = logging.getLogger("mp147.api")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)       # stdout, never a file
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


# ---------- request schema (the 2 client rules live here) ----------
class PredictRequest(BaseModel):
    drivers_nearby: int = Field(..., ge=0)
    distance_to_driver_km: float = Field(..., ge=0)   # client rule 2
    hour: int = Field(..., ge=0, le=23)
    rain_mm: float = Field(..., ge=0, le=85)          # client rule 1
    surge: float
    road_closures: int = Field(..., ge=0)
    car_type: str
    zone: str


# ---------- model loading ----------
def make_dummy_model():
    X = pd.DataFrame(
        [[1, 1.0, 8, 0.0, 1.0, 0, "mini", "Airport"],
         [2, 2.0, 9, 1.0, 1.2, 1, "sedan", "Old City"]],
        columns=FEATURES,
    )
    model = Pipeline([("reg", DummyRegressor(strategy="constant", constant=12.0))])
    model.fit(X, [12.0, 12.0])
    return model


def model_path_from_params():
    params_file = ROOT / "params.yaml"
    if not params_file.exists():
        return None
    with open(params_file) as f:
        params = yaml.safe_load(f)
    return ROOT / params["model"]["model_path"]


def load_model():
    path = os.environ.get("MODEL_PATH")                # 1) env var (tests use this)
    if path and os.path.exists(path):
        logger.info("model loaded", extra={"fields": {"source": "MODEL_PATH"}})
        return joblib.load(path)
    path = model_path_from_params()                    # 2) path from params.yaml
    if path is not None and path.exists():
        logger.info("model loaded", extra={"fields": {"source": "params.yaml"}})
        return joblib.load(path)
    logger.warning("model file not found, using dummy model")
    return make_dummy_model()                          # 3) dummy fallback


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(title="MP147 Pickup ETA API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    data = req.model_dump()
    row = pd.DataFrame([data], columns=FEATURES)
    prediction = float(app.state.model.predict(row)[0])
    logger.info("prediction", extra={"fields": {**data, "prediction": prediction}})
    return {"prediction": prediction}