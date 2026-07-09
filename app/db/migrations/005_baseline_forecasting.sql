CREATE TABLE IF NOT EXISTS baseline_predictions (
    id BIGSERIAL PRIMARY KEY,
    "timestamp" TIMESTAMPTZ NOT NULL,
    model_name TEXT NOT NULL,
    prediction NUMERIC NOT NULL,
    actual NUMERIC NOT NULL,
    error NUMERIC,
    absolute_error NUMERIC,
    percentage_error NUMERIC,
    evaluation_run_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE ("timestamp", model_name, evaluation_run_id)
);

CREATE INDEX IF NOT EXISTS ix_baseline_predictions_run_model_timestamp
    ON baseline_predictions (
        evaluation_run_id,
        model_name,
        "timestamp" DESC
    );

CREATE TABLE IF NOT EXISTS baseline_metrics (
    id BIGSERIAL PRIMARY KEY,
    evaluation_run_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    start_timestamp TIMESTAMPTZ,
    end_timestamp TIMESTAMPTZ,
    mae NUMERIC,
    rmse NUMERIC,
    mape NUMERIC,
    smape NUMERIC,
    r2 NUMERIC,
    count INTEGER,
    mean_actual NUMERIC,
    mean_prediction NUMERIC,
    max_error NUMERIC,
    median_absolute_error NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (evaluation_run_id, model_name)
);

CREATE INDEX IF NOT EXISTS ix_baseline_metrics_created_at
    ON baseline_metrics (created_at DESC);

CREATE INDEX IF NOT EXISTS ix_baseline_metrics_model_created_at
    ON baseline_metrics (model_name, created_at DESC);
