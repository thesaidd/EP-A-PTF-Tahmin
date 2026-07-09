CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS raw_epias_responses (
    id BIGSERIAL NOT NULL,
    endpoint_name VARCHAR(255) NOT NULL,
    endpoint_url TEXT NOT NULL,
    request_payload JSONB NOT NULL,
    response_json JSONB NOT NULL,
    status_code INTEGER NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_start_date DATE,
    data_end_date DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, fetched_at),
    CHECK (
        data_start_date IS NULL
        OR data_end_date IS NULL
        OR data_start_date <= data_end_date
    )
);

SELECT create_hypertable(
    'raw_epias_responses',
    'fetched_at',
    if_not_exists => TRUE
);

CREATE TABLE IF NOT EXISTS ptf_hourly (
    "timestamp" TIMESTAMPTZ NOT NULL,
    ptf_tl NUMERIC NOT NULL,
    ptf_usd NUMERIC,
    ptf_eur NUMERIC,
    source TEXT NOT NULL DEFAULT 'epias',
    raw_record JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY ("timestamp")
);

SELECT create_hypertable(
    'ptf_hourly',
    'timestamp',
    if_not_exists => TRUE
);

CREATE TABLE IF NOT EXISTS features_ptf_hourly (
    delivery_time TIMESTAMPTZ NOT NULL,
    feature_set_version VARCHAR(64) NOT NULL,
    features JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (delivery_time, feature_set_version)
);

SELECT create_hypertable(
    'features_ptf_hourly',
    'delivery_time',
    if_not_exists => TRUE
);

CREATE TABLE IF NOT EXISTS model_predictions (
    forecast_time TIMESTAMPTZ NOT NULL,
    delivery_time TIMESTAMPTZ NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    model_version VARCHAR(128) NOT NULL,
    point_forecast DOUBLE PRECISION NOT NULL,
    lower_bound DOUBLE PRECISION,
    upper_bound DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (
        forecast_time,
        delivery_time,
        model_name,
        model_version
    ),
    CHECK (lower_bound IS NULL OR upper_bound IS NULL OR lower_bound <= upper_bound)
);

SELECT create_hypertable(
    'model_predictions',
    'delivery_time',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS ix_model_predictions_lookup
    ON model_predictions (model_name, model_version, delivery_time DESC);

CREATE TABLE IF NOT EXISTS model_metrics (
    evaluation_time TIMESTAMPTZ NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    model_version VARCHAR(128) NOT NULL,
    metric_name VARCHAR(128) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    evaluation_window_start TIMESTAMPTZ,
    evaluation_window_end TIMESTAMPTZ,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (
        evaluation_time,
        model_name,
        model_version,
        metric_name
    )
);

SELECT create_hypertable(
    'model_metrics',
    'evaluation_time',
    if_not_exists => TRUE
);
