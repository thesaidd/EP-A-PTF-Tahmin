from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.models import (
    BaselineEvaluationRequest,
    BaselineEvaluationSummary,
    BaselineStatusResponse,
    ForecastDecisionRunRequest,
    ForecastDecisionStatusResponse,
    ForecastDecisionSummary,
    GprResidualStatusResponse,
    GprResidualTrainingRequest,
    GprResidualTrainingSummary,
    XGBoostStatusResponse,
    XGBoostTrainingRequest,
    XGBoostTrainingSummary,
)
from ml.models.baseline_ptf import BaselinePtfService
from ml.models.forecast_decision_ptf import ForecastDecisionPtfService
from ml.models.gpr_residual_ptf import GprResidualPtfService
from ml.models.xgboost_ptf import XGBoostPtfService

router = APIRouter(prefix="/api/models", tags=["models"])


def get_baseline_ptf_service() -> BaselinePtfService:
    return BaselinePtfService()


def get_xgboost_ptf_service() -> XGBoostPtfService:
    return XGBoostPtfService()


def get_gpr_residual_ptf_service() -> GprResidualPtfService:
    return GprResidualPtfService()


def get_forecast_decision_ptf_service() -> ForecastDecisionPtfService:
    return ForecastDecisionPtfService()


@router.post("/baseline/ptf/run", response_model=BaselineEvaluationSummary)
def run_baseline_ptf_evaluation(
    request: BaselineEvaluationRequest,
    service: BaselinePtfService = Depends(get_baseline_ptf_service),
) -> BaselineEvaluationSummary:
    try:
        summary = service.run_baseline_evaluation(
            start_date=request.start_date,
            end_date=request.end_date,
        )
        return BaselineEvaluationSummary.model_validate(summary)
    except (SQLAlchemyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get("/baseline/ptf/status", response_model=BaselineStatusResponse)
def baseline_ptf_status(
    service: BaselinePtfService = Depends(get_baseline_ptf_service),
) -> BaselineStatusResponse:
    try:
        return BaselineStatusResponse.model_validate(service.get_status())
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not query baseline evaluation status.",
        ) from exc


@router.post("/xgboost/ptf/train", response_model=XGBoostTrainingSummary)
def train_xgboost_ptf(
    request: XGBoostTrainingRequest,
    service: XGBoostPtfService = Depends(get_xgboost_ptf_service),
) -> XGBoostTrainingSummary:
    try:
        summary = service.run_training(
            train_start=request.train_start,
            train_end=request.train_end,
            test_start=request.test_start,
            test_end=request.test_end,
            model_version=request.model_version,
            feature_version=request.feature_version,
        )
        return XGBoostTrainingSummary.model_validate(summary)
    except (RuntimeError, SQLAlchemyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get("/xgboost/ptf/status", response_model=XGBoostStatusResponse)
def xgboost_ptf_status(
    service: XGBoostPtfService = Depends(get_xgboost_ptf_service),
) -> XGBoostStatusResponse:
    try:
        return XGBoostStatusResponse.model_validate(service.get_status())
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not query XGBoost training status.",
        ) from exc


@router.post(
    "/gpr-residual/ptf/train",
    response_model=GprResidualTrainingSummary,
)
def train_gpr_residual_ptf(
    request: GprResidualTrainingRequest,
    service: GprResidualPtfService = Depends(get_gpr_residual_ptf_service),
) -> GprResidualTrainingSummary:
    try:
        summary = service.run_residual_modeling(
            xgboost_training_run_id=request.xgboost_training_run_id,
            residual_train_start=request.residual_train_start,
            residual_train_end=request.residual_train_end,
            residual_test_start=request.residual_test_start,
            residual_test_end=request.residual_test_end,
            model_version=request.model_version,
            max_train_rows=request.max_train_rows,
        )
        return GprResidualTrainingSummary.model_validate(summary)
    except (RuntimeError, SQLAlchemyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get(
    "/gpr-residual/ptf/status",
    response_model=GprResidualStatusResponse,
)
def gpr_residual_ptf_status(
    service: GprResidualPtfService = Depends(get_gpr_residual_ptf_service),
) -> GprResidualStatusResponse:
    try:
        return GprResidualStatusResponse.model_validate(service.get_status())
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not query GPR residual modeling status.",
        ) from exc


@router.post(
    "/forecast-decision/ptf/run",
    response_model=ForecastDecisionSummary,
)
def run_forecast_decision_ptf(
    request: ForecastDecisionRunRequest,
    service: ForecastDecisionPtfService = Depends(
        get_forecast_decision_ptf_service
    ),
) -> ForecastDecisionSummary:
    try:
        summary = service.run_decision_layer(
            gpr_run_id=request.gpr_run_id,
            model_version=request.model_version,
        )
        return ForecastDecisionSummary.model_validate(summary)
    except (SQLAlchemyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get(
    "/forecast-decision/ptf/status",
    response_model=ForecastDecisionStatusResponse,
)
def forecast_decision_ptf_status(
    service: ForecastDecisionPtfService = Depends(
        get_forecast_decision_ptf_service
    ),
) -> ForecastDecisionStatusResponse:
    try:
        return ForecastDecisionStatusResponse.model_validate(service.get_status())
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not query forecast decision status.",
        ) from exc
