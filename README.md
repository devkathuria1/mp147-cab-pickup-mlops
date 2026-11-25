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

TODO: to be completed after the code is final.

- Setup
- Pull the data with DVC
- Train
- Run the API
- Run the tests
- Build and run Docker

## Results

TODO: to be completed after the final model is chosen.