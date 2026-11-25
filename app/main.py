import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

FEATURES = [
    "drivers_nearby", "distance_to_driver_km", "hour", "rain_mm",
    "surge", "road_closures", "car_type", "zone",
]


class PredictRequest(BaseModel):
    drivers_nearby: int = Field(..., ge=0)
    distance_to_driver_km: float = Field(..., ge=0)   # client rule 2
    hour: int = Field(..., ge=0, le=23)
    rain_mm: float = Field(..., ge=0, le=85)          # client rule 1
    surge: float
    road_closures: int = Field(..., ge=0)
    car_type: str
    zone: str


def make_dummy_model():
    """Temporary model: always predicts 12 minutes. Replaced by Vedika's model on Thursday."""
    X = pd.DataFrame(
        [[1, 1.0, 8, 0.0, 1.0, 0, "mini", "Airport"],
         [2, 2.0, 9, 1.0, 1.2, 1, "sedan", "Old City"]],
        columns=FEATURES,
    )
    model = Pipeline([("reg", DummyRegressor(strategy="constant", constant=12.0))])
    model.fit(X, [12.0, 12.0])
    return model


def load_model():
    path = os.environ.get("MODEL_PATH")
    if path and os.path.exists(path):
        return joblib.load(path)
    return make_dummy_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()   # runs once when the server starts
    yield


app = FastAPI(title="MP147 Pickup ETA API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    row = pd.DataFrame([req.model_dump()], columns=FEATURES)
    pred = app.state.model.predict(row)
    return {"prediction": float(pred[0])}