from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


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


class XGBoostTrainingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    train_start: date | None = None
    train_end: date | None = None
    test_start: date | None = None
    test_end: date | None = None
    model_version: str = "xgboost_v1"
    feature_version: str = "v1"

    @model_validator(mode="after")
    def validate_date_ranges(self) -> "XGBoostTrainingRequest":
        if (
            self.train_start is not None
            and self.train_end is not None
            and self.train_end < self.train_start
        ):
            raise ValueError("train_end must be on or after train_start")
        if (
            self.test_start is not None
            and self.test_end is not None
            and self.test_end < self.test_start
        ):
            raise ValueError("test_end must be on or after test_start")
        return self


class XGBoostTrainingSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    training_run_id: str
    model_version: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str | None
    train_rows: int
    test_rows: int
    metrics: BaselineMetricValues | dict[str, Any]
    baseline_comparison: dict[str, Any]
    artifact_path: str | None
    warnings: list[str]
    errors: list[str]


class XGBoostStatusResponse(BaseModel):
    total_prediction_rows: int
    total_metric_rows: int
    latest_training_run_id: str | None
    latest_created_at: datetime | None
    available_model_versions: list[str]
    latest_metrics: BaselineMetricValues | None
    latest_baseline_comparison: dict[str, Any] | None


class UncertaintyMetricValues(BaseModel):
    interval_coverage_95: float | None
    mean_interval_width: float | None
    median_interval_width: float | None
    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int


class GprResidualTrainingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    xgboost_training_run_id: str | None = None
    residual_train_start: date | None = None
    residual_train_end: date | None = None
    residual_test_start: date | None = None
    residual_test_end: date | None = None
    model_version: str = "gpr_residual_v1"
    max_train_rows: int = 3000

    @model_validator(mode="after")
    def validate_request(self) -> "GprResidualTrainingRequest":
        if self.max_train_rows <= 0:
            raise ValueError("max_train_rows must be positive")
        if (
            self.residual_train_start is not None
            and self.residual_train_end is not None
            and self.residual_train_end < self.residual_train_start
        ):
            raise ValueError(
                "residual_train_end must be on or after residual_train_start"
            )
        if (
            self.residual_test_start is not None
            and self.residual_test_end is not None
            and self.residual_test_end < self.residual_test_start
        ):
            raise ValueError(
                "residual_test_end must be on or after residual_test_start"
            )
        return self


class GprResidualTrainingSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    gpr_run_id: str
    xgboost_training_run_id: str
    model_version: str
    residual_train_start: str
    residual_train_end: str
    residual_test_start: str
    residual_test_end: str
    train_rows: int
    test_rows: int
    max_train_rows: int
    metrics: BaselineMetricValues | dict[str, Any]
    uncertainty_metrics: UncertaintyMetricValues | dict[str, Any]
    xgboost_comparison: dict[str, Any]
    baseline_comparison: dict[str, Any]
    artifact_path: str | None
    warnings: list[str]
    errors: list[str]


class GprResidualStatusResponse(BaseModel):
    total_prediction_rows: int
    total_metric_rows: int
    latest_gpr_run_id: str | None
    latest_created_at: datetime | None
    available_model_versions: list[str]
    latest_metrics: BaselineMetricValues | None
    latest_uncertainty_metrics: UncertaintyMetricValues | None
    latest_xgboost_comparison: dict[str, Any] | None
    latest_baseline_comparison: dict[str, Any] | None
    latest_artifact_path: str | None


class ForecastDecisionRunRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    gpr_run_id: str | None = None
    model_version: str = "forecast_decision_v1"


class ForecastDecisionSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    decision_run_id: str
    gpr_run_id: str
    xgboost_training_run_id: str
    model_version: str
    selected_model: str
    selection_reason: str
    evaluation_start: str | None
    evaluation_end: str | None
    metrics: BaselineMetricValues | dict[str, Any]
    uncertainty_metrics: UncertaintyMetricValues | dict[str, Any]
    xgboost_comparison: dict[str, Any]
    gpr_comparison: dict[str, Any]
    warnings: list[str]
    errors: list[str]


class ForecastDecisionStatusResponse(BaseModel):
    total_prediction_rows: int
    total_metric_rows: int
    latest_decision_run_id: str | None
    latest_created_at: datetime | None
    available_model_versions: list[str]
    latest_selected_model: str | None
    latest_selection_reason: str | None
    latest_metrics: BaselineMetricValues | None
    latest_uncertainty_metrics: UncertaintyMetricValues | None
    latest_xgboost_comparison: dict[str, Any] | None
    latest_gpr_comparison: dict[str, Any] | None
