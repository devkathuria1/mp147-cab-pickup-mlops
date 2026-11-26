# mp147-cab-pickup-mlops

## Cab Pick-up Time (Patna)

| | |
|---|---|
| **Group ID** | G103 |
| **Problem ID** | MP147 |
| **Course** | SCSE3040 MLOps Mini Project |
| **Faculty** | Gaurav Tripathi |
| **Batch** | 84 |
| **Client** | Patna Rides Cooperative |

## Team

| Enrolment No. | Name | Role |
|---|---|---|
| S24CSEU2190 | Devanshu Kathuria | Data, monitoring and retraining |
| S24CSEU2193 | Vedika Chhabra | Model (v1) and MLflow |
| S24CSEU2213 | Devender Singh Shekhawat  | API, tests and CI |
| S24CSEU2210 | Mahatva Varshney | Packaging, Docker, Kubernetes, predictions |

## Problem

A cab app in Patna shows riders when the cab will arrive, but that estimate is
often wrong. We build a machine learning model that predicts `pickup_eta_min`,
the minutes from booking to pick-up. This is a regression problem and we judge
it with MAE (mean absolute error), the average mistake in minutes.

## Pipeline

1. **Data**: load the messy `train.csv`, find the 7 data problems and clean them.
2. **Train**: build a scikit-learn pipeline and train several models.
3. **Track**: log every run in MLflow and register the best model as version 1.
4. **Serve**: a FastAPI service with `/health` and `/predict`.
5. **Package**: Docker image and Kubernetes files.
6. **Test and CI**: pytest tests that run on GitHub Actions.
7. **Monitor**: drift check (PSI) and error per zone on `later.csv`.
8. **Retrain**: train v2 and promote it only if it is at least 12% better.
9. **Predict**: predict the September holdout with the champion model.

## Project structure

```
src/mp147/      data, features, training, monitoring and prediction code
app/            FastAPI service
tests/          pytest tests
data/raw/       CSV data (tracked by DVC, not by git)
models/         saved model
k8s/            Kubernetes files
reports/        findings.yaml
predictions/    holdout predictions
```

## How to run

You need Python 3.12 and Git. Docker is optional.

**1. Setup**

```
git clone https://github.com/devkathuria1/mp147-cab-pickup-mlops.git
cd mp147-cab-pickup-mlops
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Mac or Linux, activate with `source .venv/bin/activate`.

**2. Data**

The CSV files are not stored in git. Copy `train.csv` and `later.csv` from the
course zip into `data/raw/` (`holdout_features.csv` is already in the repo).
Do not open or save the CSVs in Excel, because that changes the file hash.
Check that your copy is the same file:

```
python -c "import hashlib;print(hashlib.sha256(open('data/raw/train.csv','rb').read()).hexdigest())"
```

The result must be `af1ed189138a3efdaa0e2cf12765ce7325f012b468a748649ed958cc3626cef2`.

**3. Train**

```
$env:PYTHONPATH="src"; python -m mp147.train
```

On Mac or Linux: `PYTHONPATH=src python -m mp147.train`. Training logs runs to
`mlflow.db`. The v1 and v2 models are already registered there, so running it
again adds new runs. Do not commit `mlflow.db` afterwards.

**4. Predict the September holdout**

```
$env:PYTHONPATH="src"; python src/mp147/predict.py
```

This writes `predictions/holdout_predictions.csv` (250 rows) using the champion model.

**5. Run the API**

```
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/health (it returns `{"status":"ok"}`) or
http://localhost:8000/docs to try `/predict`.

**6. Run the tests**

```
python -m pytest -v
```

**7. Docker**

```
docker build -t mp147-api:1.0 .
docker run -p 8000:8000 mp147-api:1.0
```

**8. Kubernetes**

The manifests are in `k8s/` (`deployment.yaml` with 4 replicas and
`service.yaml` on port 80). With a cluster available:
`kubectl apply -f k8s/`.
## Results

**Drift (PSI, cleaned train vs later.csv).** `rain_mm` drifted the most
(PSI 3.35, a large change). All other features are below 0.1, a small change:
drivers_nearby 0.0498, distance_to_driver_km 0.022, hour 0.0296, surge 0.04,
road_closures 0.0174.

**Error by zone (v1 model on later.csv).**

| Zone | MAE (minutes) |
|---|---|
| Airport | 1.0749 |
| Old City | 1.2792 |
| Tech Park | 9.5173 |

Tech Park is the worst-served zone.

**Retraining (last 30% of later.csv, same rows for both models).**

| Model | MAE (minutes) |
|---|---|
| v1 | 3.7353 |
| v2 (train + first 70% of later) | 1.2991 |

The promotion rule is `mae_v2 <= 0.88 x mae_v1`. v2 is about 65% lower than v1,
so the decision is **promote**: version 2 is now the `champion` model.

**Holdout predictions.** `predictions/holdout_predictions.csv` has 250 rows
(`record_id,prediction`), made with the champion model.

All values are recorded in `reports/findings.yaml`.