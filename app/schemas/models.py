from datetime import date, datetime

from pydantic import BaseModel, model_validator


class BaselineEvaluationRequest(BaseModel):
    start_date: date | None = None
    end_date: date | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> "BaselineEvaluationRequest":
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.end_date < self.start_date
        ):
            raise ValueError("end_date must be on or after start_date")
        return self


class BaselineMetricValues(BaseModel):
    mae: float | None
    rmse: float | None
    mape: float | None
    smape: float | None
    r2: float | None
    count: int
    mean_actual: float | None
    mean_prediction: float | None
    max_error: float | None
    median_absolute_error: float | None


class BaselineEvaluationSummary(BaseModel):
    evaluation_run_id: str
    start_date: str
    end_date: str | None
    models_evaluated: list[str]
    metrics: dict[str, BaselineMetricValues]
    warnings: list[str]
    errors: list[str]


class BaselineStatusResponse(BaseModel):
    total_prediction_rows: int
    total_metric_rows: int
    latest_evaluation_run_id: str | None
    latest_created_at: datetime | None
    available_models: list[str]
    latest_metrics: dict[str, BaselineMetricValues]

