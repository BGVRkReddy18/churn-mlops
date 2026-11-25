# Customer Churn Prediction: End-to-End MLOps

[![CI/CD](https://github.com/BGVRkReddy18/churn-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/BGVRkReddy18/churn-mlops/actions/workflows/ci-cd.yml)

Predicts which telecom customers are about to leave, and covers the full ML lifecycle:
**versioned data → tuned & tracked training → quality-gated model registry → containerised API →
monitoring & drift detection → CI/CD**.

```
              ┌──────────── DVC pipeline (params.yaml) ────────────┐
 IBM Telco    │ ingest → prepare/validate → train (Optuna) → evaluate │──► MLflow tracking
 CSV ────────►│                                          quality gate │    + Model Registry (@champion)
              └───────────────────────────────┬───────────────────────┘
                                              │ models/model.joblib + metadata.json
                                              ▼
 GitHub Actions: lint → test → dvc repro → docker build → smoke test → push GHCR → deploy
                                              │
                                              ▼
            ┌──────── FastAPI (Docker) ────────┐   /metrics   ┌────────────┐     ┌─────────┐
 client ───►│ /predict  /predict/batch         │─────────────►│ Prometheus │────►│ Grafana │
            │ /model-info  /health             │              └────────────┘     └─────────┘
            └───────────────┬──────────────────┘
                            │ logs/predictions.jsonl
                            ▼
                   drift check (PSI vs training data)
```

## Results (test set, 1,409 customers)

| Metric | Value |
|---|---|
| ROC-AUC | 0.846 |
| PR-AUC | 0.660 |
| Recall (churners caught) | 77.8% |
| Precision | 52.6% |
| F1 | 0.628 |

The decision threshold (0.29 rather than 0.5) is tuned for F1 using out-of-fold predictions on the
training set, so the test set never influences it.

## Tech stack

| Concern | Tool |
|---|---|
| Pipeline orchestration & data versioning | **DVC** (`dvc.yaml`, `params.yaml`) |
| Modelling | scikit-learn Pipelines (LogReg / RandomForest / HistGradientBoosting) |
| Hyperparameter tuning | **Optuna** (TPE, stratified 5-fold CV) |
| Experiment tracking & model registry | **MLflow** (nested runs per trial, `@champion` alias) |
| Data validation | Schema checks in `churn.data.prepare` (fail fast on bad data) |
| Serving | **FastAPI** + Pydantic (strict typed inputs) |
| Packaging | **Docker** (multi-stage, non-root, healthcheck) + Docker Compose |
| Monitoring | **Prometheus** + **Grafana** dashboard, PSI drift detection |
| CI/CD | **GitHub Actions** → GHCR → Render (optional) |
| Quality | pytest (24 tests), ruff |

## Project structure

```
├── params.yaml                  # all tunable parameters
├── dvc.yaml                     # pipeline definition (4 stages)
├── src/churn/
│   ├── schema.py                # single source of truth for features & allowed values
│   ├── config.py                # paths + params loading
│   ├── data/ingest.py           # stage 1: download raw data
│   ├── data/prepare.py          # stage 2: clean, validate, stratified split
│   ├── features.py              # preprocessing + model factory + Optuna search spaces
│   ├── train.py                 # stage 3: tune, choose threshold, fit, log to MLflow
│   ├── evaluate.py              # stage 4: test metrics, plots, quality gate, registry
│   ├── api/                     # FastAPI app, schemas, model service
│   └── monitoring/drift.py      # PSI-based data drift report
├── tests/                       # data, features, API and drift tests
├── scripts/simulate_traffic.py  # generate normal or drifted traffic
├── docker/api.Dockerfile
├── docker-compose.yml           # api + prometheus + grafana
├── monitoring/                  # prometheus config, grafana provisioning + dashboard
└── .github/workflows/ci-cd.yml
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
make install

make pipeline        # run ingest → prepare → train → evaluate
make mlflow          # experiments & registry   → http://localhost:5050
make test            # tests + coverage

make up              # Docker: API + Prometheus + Grafana
#   Web page   → http://localhost:8080/   (project page + live prediction form)
#   API docs   → http://localhost:8080/docs
#   Prometheus → http://localhost:9091
#   Grafana    → http://localhost:3001  (dashboard "Churn API")

make simulate        # send normal traffic   → watch Grafana
make drift           # no drift expected
make simulate-drift  # send shifted traffic (higher prices, new month-to-month customers)
make drift           # tenure, Contract, MonthlyCharges, TotalCharges flagged
make down
```

Run `make help` to list all commands.

### Example request

```bash
curl -X POST localhost:8080/predict -H 'Content-Type: application/json' -d '{
  "gender":"Female","SeniorCitizen":"No","Partner":"Yes","Dependents":"No","tenure":2,
  "PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
  "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
  "StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month",
  "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
  "MonthlyCharges":95.5,"TotalCharges":191.0}'
# {"churn":true,"churn_probability":0.7782,"threshold":0.2939,"model_version":"1"}
```

## How the pieces fit

- **Reproducibility.** `dvc repro` re-runs only the stages whose code, data or params changed.
  Try it: change `train.model` to `random_forest` in `params.yaml`, run `make pipeline`, then
  `dvc metrics diff`.
- **No train/serve skew.** Preprocessing and model are a single sklearn `Pipeline`, and the API
  imports the same `schema.py` that the validation code uses.
- **Quality gate and promotion.** `evaluate` fails the pipeline (and CI) if test ROC-AUC is below
  `evaluate.min_roc_auc`. Passing models are registered in MLflow; one becomes `@champion` only if
  it scores at least as well as the current champion.
- **Monitoring.** The API exports request rate, latency, error rate, prediction mix and probability
  distribution to Prometheus. Each prediction's inputs are logged to JSONL, and `make drift`
  compares them to the training data with PSI (> 0.2 = drift; `--fail-on-drift` exits with code 1
  so a cron job or CI can trigger retraining).

## Research: does drift-triggered retraining actually help?

`src/driftlab/` is a separate, DVC-tracked study (`research/dvc.yaml`) that tests the assumption
behind this system's monitoring: *input drift alarm → retrain*. It covers 7 drift detectors, 10
retraining policies, 4 injected drift types on the Telco data with 50 runs each, and Elec2 as a
dataset with real drift. Detectors are calibrated to an equal false-alarm rate, and results are
tested with Friedman/Nemenyi and Holm-corrected Wilcoxon tests.

Headline results (full write-up: [`research/REPORT.md`](research/REPORT.md)):

- The only harmful drift (concept drift, −0.33 AUC) is **invisible** to all input detectors,
  including PSI. They catch harmless covariate shift instead.
- On Telco, retraining on input-drift alarms is **worse than never retraining**. Triggering on a
  labelled AUC drop matches the oracle and uses 77 % fewer retrains than a fixed schedule.
- On Elec2, where drift is in the inputs, a calibrated KS trigger matches retraining every week with
  64 % fewer retrains.

```bash
make research          # ~70 min; make research-quick for a 5-minute smoke run
```

## CI/CD

`.github/workflows/ci-cd.yml` runs on every push and PR:

1. **test**: ruff lint/format, pytest with coverage
2. **train**: `dvc repro` (the quality gate can fail the build) and a metrics table in the job summary
3. **docker**: build image, smoke-test `/health` and `/predict`, push `ghcr.io/<repo>/churn-api` on `main`
4. **deploy**: triggers a Render deploy hook if the `RENDER_DEPLOY_HOOK_URL` secret is set

Optional secret `MLFLOW_TRACKING_URI` sends CI runs to a shared MLflow server.

## Deploy (free, on Render)

`render.yaml` is a Render Blueprint, so deployment takes one click:

1. Sign in at [render.com](https://render.com) with GitHub.
2. **New → Blueprint**, pick this repository, then **Apply**.
3. Render builds `docker/api.Dockerfile` and serves the project page at
   `https://<service-name>.onrender.com/` (API docs at `/docs`).

After that, every push to `main` redeploys automatically, but only after GitHub Actions passes. On
the free plan the service sleeps after about 15 minutes idle, and the first request then takes
roughly 30–60 seconds.

## Ideas to extend

- DVC remote (S3 / GCS) so data and models are shared via `dvc push` / `dvc pull`
- Scheduled GitHub Action: labelled-performance check → retrain → quality gate (input drift alone is a poor trigger, see `research/REPORT.md`)
- Serve straight from the registry (`models:/churn-classifier@champion`)
- SHAP explanations per prediction
- Deploy to AWS ECS / GCP Cloud Run with Terraform
