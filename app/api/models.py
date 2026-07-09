from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.models import (
    BaselineEvaluationRequest,
    BaselineEvaluationSummary,
    BaselineStatusResponse,
)
from ml.models.baseline_ptf import BaselinePtfService

router = APIRouter(prefix="/api/models", tags=["models"])


def get_baseline_ptf_service() -> BaselinePtfService:
    return BaselinePtfService()


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

