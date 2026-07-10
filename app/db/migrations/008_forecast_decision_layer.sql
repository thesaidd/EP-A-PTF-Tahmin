CREATE TABLE IF NOT EXISTS forecast_decision_predictions (
    id BIGSERIAL PRIMARY KEY,
    "timestamp" TIMESTAMPTZ NOT NULL,
    decision_run_id TEXT NOT NULL,
    gpr_run_id TEXT NOT NULL,
    xgboost_training_run_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    selected_model TEXT NOT NULL,
    xgboost_prediction NUMERIC NOT NULL,
    gpr_corrected_prediction NUMERIC NOT NULL,
    selected_prediction NUMERIC NOT NULL,
    actual NUMERIC NOT NULL,
    residual_mean NUMERIC,
    residual_std NUMERIC NOT NULL,
    lower_bound_95 NUMERIC NOT NULL,
    upper_bound_95 NUMERIC NOT NULL,
    interval_width_95 NUMERIC NOT NULL,
    risk_level TEXT NOT NULL,
    error NUMERIC,
    absolute_error NUMERIC,
    percentage_error NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE ("timestamp", decision_run_id)
);

CREATE INDEX IF NOT EXISTS ix_forecast_decision_predictions_run_timestamp
    ON forecast_decision_predictions (decision_run_id, "timestamp");

CREATE INDEX IF NOT EXISTS ix_forecast_decision_predictions_gpr_run
    ON forecast_decision_predictions (gpr_run_id);

CREATE TABLE IF NOT EXISTS forecast_decision_metrics (
    id BIGSERIAL PRIMARY KEY,
    decision_run_id TEXT NOT NULL,
    gpr_run_id TEXT NOT NULL,
    xgboost_training_run_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    selected_model TEXT NOT NULL,
    selection_reason TEXT,
    evaluation_start TIMESTAMPTZ,
    evaluation_end TIMESTAMPTZ,
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
    interval_coverage_95 NUMERIC,
    mean_interval_width NUMERIC,
    median_interval_width NUMERIC,
    low_risk_count INTEGER,
    medium_risk_count INTEGER,
    high_risk_count INTEGER,
    xgboost_comparison JSONB,
    gpr_comparison JSONB,
    decision_params JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (decision_run_id)
);

CREATE INDEX IF NOT EXISTS ix_forecast_decision_metrics_created_at
    ON forecast_decision_metrics (created_at DESC);

CREATE INDEX IF NOT EXISTS ix_forecast_decision_metrics_model_version
    ON forecast_decision_metrics (model_version);
