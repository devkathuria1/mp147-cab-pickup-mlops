"""features.py - the model recipe (Vedika).

Contract with the team:
  FEATURES = the 8 request fields, in this order
  TARGET   = 'pickup_eta_min'
  build_pipeline(model_name, **params) -> sklearn Pipeline
The Pipeline takes a RAW DataFrame with exactly the 8 FEATURES columns.
Only built-in sklearn classes are used, so joblib.load works in the API.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "drivers_nearby",
    "distance_to_driver_km",
    "hour",
    "rain_mm",
    "surge",
    "road_closures",
]
CATEGORICAL_FEATURES = ["car_type", "zone"]

# Order matters: this is the order of the 8 request fields in the contract.
FEATURES = [
    "drivers_nearby",
    "distance_to_driver_km",
    "hour",
    "rain_mm",
    "surge",
    "road_closures",
    "car_type",
    "zone",
]
TARGET = "pickup_eta_min"

MODELS = {
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "ridge": Ridge,
}


def build_pipeline(model_name, **params):
    """Return an unfitted Pipeline: preprocessing + the chosen regressor."""
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Choose from {list(MODELS)}"
        )

    numeric_part = SimpleImputer(strategy="median")

    categorical_part = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_part, NUMERIC_FEATURES),
            ("cat", categorical_part, CATEGORICAL_FEATURES),
        ]
    )

    regressor = MODELS[model_name](**params)

    return Pipeline(steps=[("preprocess", preprocess), ("model", regressor)])