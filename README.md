# EPİAŞ PTF Forecasting MVP

A production-oriented starting point for a B2B energy forecasting service for
Turkey's Day-Ahead Market. The target is an hourly 24-hour-ahead forecast of the
Market Clearing Price (PTF/MCP) using data from the EPİAŞ Transparency Platform.

This repository currently contains infrastructure and application scaffolding
only. EPİAŞ ingestion, feature engineering, XGBoost point forecasts, and
Gaussian Process residual uncertainty modeling are intentionally not yet
implemented.

## Architecture

The local stack contains:

- **FastAPI** for health checks and, later, forecast serving.
- **Streamlit** for the MVP user interface.
- **PostgreSQL with TimescaleDB** for raw, feature, prediction, and metric data.
- **MLflow** for experiment tracking and model artifacts.
- **Docker Compose** for reproducible local and cloud-oriented deployment.

The Python packages are separated by responsibility:

```text
app/             API, configuration, schemas, services, and database code
data_pipeline/   EPİAŞ adapters, ingestion jobs, and data validation
ml/              Features, models, training, inference, and evaluation
dashboard/       Streamlit MVP
scripts/         Operational and developer scripts
tests/           Automated tests
docs/            Project documentation
```

## Run with Docker Compose

Requirements:

- Docker Engine or Docker Desktop
- Docker Compose v2

Start the complete stack:

```bash
cp .env.example .env
docker compose up --build
```

On Windows PowerShell, use:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

The services are then available at:

- API: <http://localhost:8000>
- API documentation: <http://localhost:8000/docs>
- API health: <http://localhost:8000/health>
- Streamlit: <http://localhost:8501>
- MLflow: <http://localhost:5000>
- PostgreSQL: `localhost:5432`

Stop the stack with:

```bash
docker compose down
```

Use `docker compose down -v` only when you intentionally want to delete local
database and MLflow volumes.

## Environment variables

Copy `.env.example` to `.env` and adjust values as needed.

| Variable | Purpose | Default |
| --- | --- | --- |
| `PROJECT_NAME` | API/project display name | `EPİAŞ PTF Forecasting MVP` |
| `APP_VERSION` | Application version returned by the API | `0.1.0` |
| `ENVIRONMENT` | Runtime environment label | `development` |
| `POSTGRES_USER` | PostgreSQL user | `pepias` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `pepias` |
| `POSTGRES_DB` | PostgreSQL database | `pepias` |
| `DATABASE_URL` | SQLAlchemy-compatible database URL | Compose database URL |
| `MLFLOW_DATABASE` | Dedicated PostgreSQL database for MLflow metadata | `mlflow` |
| `POSTGRES_PORT` | Host PostgreSQL port | `5432` |
| `API_PORT` | Host FastAPI port | `8000` |
| `STREAMLIT_PORT` | Host Streamlit port | `8501` |
| `MLFLOW_PORT` | Host MLflow port | `5000` |
| `API_URL` | Internal API URL used by Streamlit | `http://api:8000` |
| `MLFLOW_TRACKING_URI` | MLflow tracking URI used by application code | `http://mlflow:5000` |
| `MLFLOW_BACKEND_STORE_URI` | Dedicated PostgreSQL URI used for MLflow metadata | `postgresql+psycopg2://pepias:pepias@db:5432/mlflow` |
| `EPIAS_BASE_URL` | EPİAŞ Transparency Platform base URL | `https://seffaflik.epias.com.tr` |
| `EPIAS_AUTH_URL` | EPİAŞ authentication service base URL | `https://giris.epias.com.tr` |
| `EPIAS_USERNAME` | EPİAŞ account username | Empty |
| `EPIAS_PASSWORD` | EPİAŞ account password | Empty |
| `EPIAS_REQUEST_TIMEOUT` | HTTP timeout in seconds | `30` |
| `EPIAS_MAX_RETRIES` | Retries after a failed HTTP attempt | `3` |
| `EPIAS_PTF_ENDPOINT` | Configurable MCP/PTF endpoint path | `/electricity-service/v1/markets/dam/data/mcp` |

Do not use the example credentials in a shared or production environment.

## EPİAŞ API Client

`data_pipeline/epias/client.py` provides a generic synchronous HTTP client for
EPİAŞ POST endpoints. It sends JSON requests, applies timeouts and bounded
exponential-backoff retries, and raises clear errors for authentication,
transport, HTTP, and JSON parsing failures.
[EPİAŞ's official technical documentation](https://seffaflik.epias.com.tr/electricity-service/technical/tr/index.html)
defines the authentication and `TGT` header contract used here.

Authenticated calls obtain a TGT from
`EPIAS_AUTH_URL/cas/v1/tickets` using form-encoded credentials. The token is
cached in memory, refreshed before its two-hour lifetime expires, and refreshed
once if an API request returns HTTP 401 or 403. Missing credentials do not stop
the API from starting; only a request with `use_auth=true` requires them.
Credentials and TGT values are never included in application logs or health
responses.

Check client configuration without making an external request:

```bash
curl http://localhost:8000/api/epias/health
```

The development-only manual POST route accepts a relative EPİAŞ endpoint:

```bash
curl -X POST http://localhost:8000/api/epias/test-post \
  -H "Content-Type: application/json" \
  -d '{"endpoint":"/electricity-service/v1/example","payload":{},"use_auth":true}'
```

Replace the example endpoint with a valid EPİAŞ POST endpoint and its required
payload. Successful responses are stored in `raw_epias_responses` with the
endpoint identity and URL, request JSON, response JSON, HTTP status, fetch
timestamp, and optional data date range.

For an existing Docker database volume created before Sprint 2, apply the
idempotent compatibility migration once:

```bash
docker compose exec db sh -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -f /docker-entrypoint-initdb.d/002_raw_epias_response_columns.sql'
```

Fresh database volumes run the initialization files automatically.

## PTF Historical Ingestion

The PTF ingestion pipeline downloads hourly Day-Ahead Market Clearing Price
data from EPİAŞ, stores every source response in `raw_epias_responses`, and
upserts normalized prices into `ptf_hourly`. Re-running the same period updates
existing timestamps instead of creating duplicates.

Create a local `.env` file from `.env.example`, then add your own EPİAŞ login:

```dotenv
EPIAS_USERNAME=your-login-email@example.com
EPIAS_PASSWORD=your-password
```

Never commit `.env`; it is already excluded by `.gitignore`. Real EPİAŞ
credentials are required for ingestion because the MCP endpoint requires a TGT.
The health and status endpoints do not require credentials.

After rebuilding the services, check the current table status:

```bash
curl http://localhost:8000/api/epias/ptf/status
```

Run a small one-day ingestion from the API container:

```bash
docker compose exec api python scripts/ingest_ptf.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-01
```

The same operation is available in FastAPI Swagger at
<http://localhost:8000/docs> through `POST /api/epias/ptf/ingest`:

```json
{
  "start_date": "2024-01-01",
  "end_date": "2024-01-01",
  "chunk_days": 30
}
```

Verify the stored row count using the PostgreSQL user and database configured
inside the Compose service:

```bash
docker compose exec db sh -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "SELECT COUNT(*) FROM ptf_hourly;"'
```

Long periods are split into bounded requests. Keep the default 30-day chunks
unless EPİAŞ operational guidance requires a smaller window; multi-year ranges
can make many requests and should be run from the CLI rather than an HTTP
request.

## PTF Feature Engineering

The feature pipeline reads normalized hourly prices from `ptf_hourly` and
upserts typed, ML-ready rows into `features_ptf_hourly`. It creates Istanbul
calendar indicators, hourly and weekly lags, rolling price statistics, and
price-change features. A range build loads 30 days of earlier source history so
the first requested timestamp can use available lag and rolling context.

Build all available PTF features:

```bash
docker compose exec api python scripts/build_ptf_features.py \
  --feature-version v1
```

Build a selected date range:

```bash
docker compose exec api python scripts/build_ptf_features.py \
  --start-date 2020-01-01 \
  --end-date 2026-07-09 \
  --feature-version v1
```

The same build is available through `POST /api/features/ptf/build` in FastAPI
Swagger at <http://localhost:8000/docs>:

```json
{
  "start_date": "2020-01-01",
  "end_date": "2026-07-09",
  "feature_version": "v1"
}
```

Check feature status:

```bash
curl http://localhost:8000/api/features/ptf/status
```

Verify the feature row count directly:

```bash
docker compose exec db sh -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "SELECT COUNT(*) FROM features_ptf_hourly;"'
```

All rolling statistics are computed from `target_ptf.shift(1)` before applying
the window. This deliberately excludes the current hour's target and prevents
target leakage into model inputs.

## Baseline PTF Forecasting

Baseline forecasts provide transparent benchmarks that future trained models
must outperform. The current evaluation compares the previous-day price,
previous-week price, trailing 24-hour mean, and trailing seven-day mean against
the observed hourly PTF. Evaluation is date-based and never randomly shuffled.

Run the default evaluation from 2024-01-01 through the latest feature timestamp:

```bash
docker compose exec api python scripts/run_baseline_ptf.py
```

Run a selected interval:

```bash
docker compose exec api python scripts/run_baseline_ptf.py \
  --start-date 2024-01-01 \
  --end-date 2026-07-09
```

The API operation is available as `POST /api/models/baseline/ptf/run` in
FastAPI Swagger at <http://localhost:8000/docs>:

```json
{
  "start_date": "2024-01-01",
  "end_date": "2026-07-09"
}
```

Inspect the latest evaluation:

```bash
curl http://localhost:8000/api/models/baseline/ptf/status
```

Inspect stored metrics directly:

```bash
docker compose exec db sh -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "SELECT model_name, mae, rmse, mape, smape, r2, count \
      FROM baseline_metrics ORDER BY created_at DESC, model_name LIMIT 20;"'
```

Each successful evaluation is also logged to the
`ptf_baseline_forecasting` MLflow experiment when MLflow is available. Database
results remain valid if MLflow is temporarily unavailable.

## XGBoost PTF Forecasting

The XGBoost point-forecasting pipeline trains an `XGBRegressor` on
`features_ptf_hourly` with a chronological train/test split. It uses numeric and
boolean feature columns, one-hot encodes `season`, drops rows missing key lag
features, excludes current-target-derived change columns to avoid leakage, and
never randomly shuffles time-series data.

Default split:

- Train: 2020-01-01 through 2023-12-31
- Test: 2024-01-01 through the latest available feature timestamp

Train from the CLI with defaults:

```bash
docker compose exec api python scripts/train_xgboost_ptf.py
```

Train a selected interval:

```bash
docker compose exec api python scripts/train_xgboost_ptf.py \
  --train-start 2020-01-01 \
  --train-end 2023-12-31 \
  --test-start 2024-01-01 \
  --test-end 2026-07-09 \
  --model-version xgboost_v1
```

The API operation is available as `POST /api/models/xgboost/ptf/train` in
FastAPI Swagger at <http://localhost:8000/docs>:

```json
{
  "train_start": "2020-01-01",
  "train_end": "2023-12-31",
  "test_start": "2024-01-01",
  "test_end": "2026-07-09",
  "model_version": "xgboost_v1",
  "feature_version": "v1"
}
```

Check training status:

```bash
curl http://localhost:8000/api/models/xgboost/ptf/status
```

Inspect stored metrics directly:

```bash
docker compose exec db sh -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "SELECT training_run_id, model_version, mae, rmse, mape, smape, r2, count, \
             baseline_comparison \
      FROM xgboost_metrics ORDER BY created_at DESC LIMIT 5;"'
```

Each successful run stores hourly test predictions in `xgboost_predictions`,
summary metrics in `xgboost_metrics`, and a native model artifact under:

```text
artifacts/models/ptf/xgboost/{model_version}/{training_run_id}/model.json
```

Artifacts are ignored by Git. Runs are also logged to the
`ptf_xgboost_forecasting` MLflow experiment when MLflow is available. If MLflow
is temporarily unavailable, training still persists database metrics and the
local model artifact.

The XGBoost summary compares MAE against the latest baseline evaluation and
returns the best baseline model plus the percentage MAE improvement. The GPR
residual layer below uses these XGBoost predictions as its point-forecast input.

## GPR Residual Uncertainty Modeling

The GPR residual model is a second-stage uncertainty layer on top of XGBoost. It
does not replace or retrain XGBoost. Instead, it learns recent XGBoost residuals:

```text
residual = actual - xgboost_prediction
```

For each evaluated timestamp, the GPR model estimates:

- `residual_mean`
- `residual_std`

The corrected point forecast is:

```text
final_prediction = xgboost_prediction + residual_mean
```

The current 95% interval is:

```text
lower_bound_95 = final_prediction - 1.96 * residual_std
upper_bound_95 = final_prediction + 1.96 * residual_std
```

Risk levels are assigned from interval width percentiles within the evaluation
set:

- `LOW`: width <= 50th percentile
- `MEDIUM`: 50th percentile < width <= 85th percentile
- `HIGH`: width > 85th percentile

Train the residual model from the CLI using the latest successful XGBoost run:

```bash
docker compose exec api python scripts/train_gpr_residual_ptf.py \
  --residual-train-start 2024-01-01 \
  --residual-train-end 2025-12-31 \
  --residual-test-start 2026-01-01 \
  --residual-test-end 2026-07-09 \
  --model-version gpr_residual_v1 \
  --max-train-rows 3000
```

Train against a specific XGBoost training run:

```bash
docker compose exec api python scripts/train_gpr_residual_ptf.py \
  --xgboost-training-run-id <xgboost-training-run-id> \
  --max-train-rows 3000
```

The API operation is available as
`POST /api/models/gpr-residual/ptf/train` in FastAPI Swagger at
<http://localhost:8000/docs>:

```json
{
  "xgboost_training_run_id": null,
  "residual_train_start": "2024-01-01",
  "residual_train_end": "2025-12-31",
  "residual_test_start": "2026-01-01",
  "residual_test_end": "2026-07-09",
  "model_version": "gpr_residual_v1",
  "max_train_rows": 3000
}
```

Check GPR residual status:

```bash
curl http://localhost:8000/api/models/gpr-residual/ptf/status
```

Inspect stored metrics:

```bash
docker compose exec db sh -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "SELECT gpr_run_id, mae, rmse, r2, interval_coverage_95, \
             mean_interval_width, xgboost_comparison, baseline_comparison, \
             artifact_path \
      FROM gpr_residual_metrics ORDER BY created_at DESC LIMIT 5;"'
```

Inspect sample predictions with intervals and risk levels:

```bash
docker compose exec db sh -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "SELECT timestamp, xgboost_prediction, residual_mean, residual_std, \
             final_prediction, lower_bound_95, upper_bound_95, risk_level \
      FROM gpr_residual_predictions \
      ORDER BY created_at DESC, timestamp LIMIT 10;"'
```

Each successful run stores interval predictions in `gpr_residual_predictions`,
summary metrics in `gpr_residual_metrics`, and a joblib artifact under:

```text
artifacts/models/ptf/gpr_residual/{model_version}/{gpr_run_id}/model.joblib
```

Artifacts are ignored by Git. Runs are also logged to the
`ptf_gpr_residual_forecasting` MLflow experiment when MLflow is available.

GPR correction is evaluated fairly against XGBoost on the same residual test
window. In early MVP runs, the uncertainty layer may improve interval/risk
visibility without improving point MAE; the `xgboost_comparison` JSON makes this
explicit.

## Forecast Decision Layer

The forecast decision layer turns model evaluation into product-safe forecast
selection. It prevents the GPR residual correction from degrading the displayed
point forecast while still preserving GPR uncertainty intervals and risk levels.

The rule is intentionally simple:

```text
if GPR corrected MAE improves over XGBoost MAE on the same evaluation window:
    selected_prediction = gpr_final_prediction
    selected_model = "gpr_corrected"
else:
    selected_prediction = xgboost_prediction
    selected_model = "xgboost"
```

Regardless of which point forecast is selected, the interval is centered around
the selected point forecast using GPR uncertainty:

```text
lower_bound_95 = selected_prediction - 1.96 * residual_std
upper_bound_95 = selected_prediction + 1.96 * residual_std
```

This means the current MVP can show XGBoost as the point forecast while still
using GPR for uncertainty and `risk_level` when GPR correction does not improve
accuracy.

Run the decision layer from the CLI using the latest successful GPR run:

```bash
docker compose exec api python scripts/run_forecast_decision_ptf.py
```

Run for a specific GPR run:

```bash
docker compose exec api python scripts/run_forecast_decision_ptf.py \
  --gpr-run-id <gpr-run-id> \
  --model-version forecast_decision_v1
```

The API operation is available as
`POST /api/models/forecast-decision/ptf/run` in FastAPI Swagger at
<http://localhost:8000/docs>:

```json
{
  "gpr_run_id": null,
  "model_version": "forecast_decision_v1"
}
```

Check decision-layer status:

```bash
curl http://localhost:8000/api/models/forecast-decision/ptf/status
```

Inspect stored decision metrics:

```bash
docker compose exec db sh -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "SELECT decision_run_id, selected_model, mae, rmse, r2, \
             interval_coverage_95, xgboost_comparison, gpr_comparison \
      FROM forecast_decision_metrics ORDER BY created_at DESC LIMIT 5;"'
```

Inspect sample selected predictions and intervals:

```bash
docker compose exec db sh -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "SELECT timestamp, selected_model, xgboost_prediction, \
             gpr_corrected_prediction, selected_prediction, lower_bound_95, \
             upper_bound_95, risk_level \
      FROM forecast_decision_predictions \
      ORDER BY created_at DESC, timestamp LIMIT 10;"'
```

For the current model findings, the expected selected point model is `xgboost`
because GPR correction did not beat XGBoost on the same residual test window.

## MLflow database separation

Application time-series tables and MLflow metadata use separate PostgreSQL
databases on the same server:

- `POSTGRES_DB` (`pepias` by default) contains application tables such as
  `raw_epias_responses` and `ptf_hourly`.
- `MLFLOW_DATABASE` (`mlflow` by default) contains only MLflow experiments,
  runs, metrics, parameters, and registry metadata.

The `db-init` Compose service creates the MLflow database idempotently, including
when an existing PostgreSQL volume is reused. The `mlflow-db-upgrade` service
then initializes a fresh MLflow schema or upgrades an existing schema before the
tracking server starts. MLflow uses the psycopg2 SQLAlchemy driver for
compatibility with the integer experiment IDs in MLflow 2.22; the FastAPI
application continues to use psycopg v3.

Existing MLflow tables in the application database are not deleted or modified.
They are retained as old local metadata but are no longer queried by the MLflow
server. A volume reset is therefore not required.

If a disposable local environment still contains incompatible state and none of
its PostgreSQL or MLflow artifact data needs to be preserved, it can be reset
explicitly:

```bash
docker compose down -v
docker compose up --build
```

This command permanently removes all local application data, MLflow metadata,
and stored artifacts. Back up or export anything important first.

## Database initialization

On the first database startup,
the initialization files under `app/db/migrations` create the dedicated MLflow
database, enable TimescaleDB, create the application tables, and convert
time-indexed tables to hypertables. Docker's PostgreSQL initialization mechanism
runs these files only when the database volume is empty; `db-init` separately
ensures the MLflow database exists for reused volumes.

## Run tests locally

With Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

PowerShell activation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

## Development roadmap

1. Implement historical PTF ingestion using the EPİAŞ client.
2. Add schema validation, data-quality checks, and scheduled jobs.
3. Build leakage-safe hourly features and reproducible training datasets.
4. Train and track an XGBoost point-forecasting model in MLflow.
5. Model residual uncertainty with Gaussian Process Regression.
6. Add a forecast decision layer for product-safe model selection.
7. Add production day-ahead forecast endpoints and model loading.
8. Expand the dashboard with forecast curves, confidence intervals, and model
   monitoring.
9. Add CI, automated migrations, observability, secrets management, and
   production deployment configuration.
