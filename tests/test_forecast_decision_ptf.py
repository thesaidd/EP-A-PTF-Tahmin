from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.models import get_forecast_decision_ptf_service
from app.main import app
from ml.models.forecast_decision_ptf import (
    ForecastDecisionPtfService,
    SELECTED_MODEL_GPR,
    SELECTED_MODEL_XGBOOST,
)

ISTANBUL = ZoneInfo("Europe/Istanbul")


def decision_source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=4,
                freq="h",
                tz=ISTANBUL,
            ),
            "gpr_run_id": ["gpr-run"] * 4,
            "xgboost_training_run_id": ["xgb-run"] * 4,
            "xgboost_prediction": [100.0, 120.0, 140.0, 160.0],
            "gpr_corrected_prediction": [90.0, 130.0, 145.0, 150.0],
            "actual": [102.0, 118.0, 150.0, 158.0],
            "residual_mean": [-10.0, 10.0, 5.0, -10.0],
            "residual_std": [5.0, 10.0, 20.0, 40.0],
            "risk_level": ["LOW", "LOW", "MEDIUM", "HIGH"],
        }
    )


def test_selected_model_is_xgboost_when_gpr_improvement_negative() -> None:
    selected_model, reason = ForecastDecisionPtfService().decide_selected_model(
        {"xgboost_comparison": {"mae_improvement_pct": -6.3}}
    )

    assert selected_model == SELECTED_MODEL_XGBOOST
    assert "did not improve" in reason


def test_selected_model_is_gpr_when_gpr_improvement_positive() -> None:
    selected_model, reason = ForecastDecisionPtfService().decide_selected_model(
        {"xgboost_comparison": {"mae_improvement_pct": 3.5}}
    )

    assert selected_model == SELECTED_MODEL_GPR
    assert "improved" in reason


def test_selected_prediction_uses_xgboost_and_recenters_intervals() -> None:
    service = ForecastDecisionPtfService()

    result = service.build_decision_predictions(
        decision_source_frame(),
        SELECTED_MODEL_XGBOOST,
    )

    assert result["selected_prediction"].tolist() == result[
        "xgboost_prediction"
    ].tolist()
    assert result["lower_bound_95"].iloc[0] == pytest.approx(100.0 - 1.96 * 5.0)
    assert result["upper_bound_95"].iloc[0] == pytest.approx(100.0 + 1.96 * 5.0)
    assert result["risk_level"].tolist() == ["LOW", "LOW", "MEDIUM", "HIGH"]


def test_selected_prediction_uses_gpr_when_selected() -> None:
    result = ForecastDecisionPtfService().build_decision_predictions(
        decision_source_frame(),
        SELECTED_MODEL_GPR,
    )

    assert result["selected_prediction"].tolist() == result[
        "gpr_corrected_prediction"
    ].tolist()


def test_decision_metrics_include_interval_coverage_and_risk_counts() -> None:
    service = ForecastDecisionPtfService()
    predictions = service.build_decision_predictions(
        decision_source_frame(),
        SELECTED_MODEL_XGBOOST,
    )

    metrics, uncertainty = service.calculate_decision_metrics(predictions)

    assert metrics["count"] == 4
    assert metrics["mae"] == pytest.approx(4.0)
    assert uncertainty["interval_coverage_95"] == pytest.approx(100.0)
    assert uncertainty["low_risk_count"] == 2
    assert uncertainty["medium_risk_count"] == 1
    assert uncertainty["high_risk_count"] == 1


def test_comparisons_are_calculated_against_same_rows() -> None:
    service = ForecastDecisionPtfService()
    predictions = service.build_decision_predictions(
        decision_source_frame(),
        SELECTED_MODEL_XGBOOST,
    )
    metrics, _ = service.calculate_decision_metrics(predictions)

    xgb_comparison, gpr_comparison = service.build_model_comparisons(
        predictions,
        metrics["mae"],
    )

    assert xgb_comparison["selected_vs_xgboost_improvement_pct"] == pytest.approx(
        0.0
    )
    assert gpr_comparison["gpr_corrected_mae"] is not None


class InMemoryForecastDecisionService(ForecastDecisionPtfService):
    def get_latest_successful_gpr_run(self) -> str | None:
        return "gpr-run"

    def load_gpr_metrics(self, gpr_run_id: str) -> dict[str, object]:
        return {
            "gpr_run_id": gpr_run_id,
            "xgboost_training_run_id": "xgb-run",
            "xgboost_comparison": {"mae_improvement_pct": -1.0},
        }

    def load_gpr_predictions(self, gpr_run_id: str) -> pd.DataFrame:
        return decision_source_frame()

    def store_decision_predictions(self, *args: object, **kwargs: object) -> int:
        return 4

    def store_decision_metrics(self, *args: object, **kwargs: object) -> int:
        return 1


def test_run_decision_layer_summary_shape() -> None:
    summary = InMemoryForecastDecisionService().run_decision_layer()

    assert summary["decision_run_id"]
    assert summary["gpr_run_id"] == "gpr-run"
    assert summary["xgboost_training_run_id"] == "xgb-run"
    assert summary["selected_model"] == SELECTED_MODEL_XGBOOST
    assert summary["metrics"]["count"] == 4
    assert summary["uncertainty_metrics"]["interval_coverage_95"] is not None
    assert summary["errors"] == []


class FakeForecastDecisionStatusService:
    def get_status(self) -> dict[str, object]:
        metric = {
            "mae": 1.0,
            "rmse": 2.0,
            "mape": 3.0,
            "smape": 4.0,
            "r2": 0.9,
            "count": 24,
            "mean_actual": 100.0,
            "mean_prediction": 99.0,
            "max_error": 5.0,
            "median_absolute_error": 1.0,
        }
        uncertainty = {
            "interval_coverage_95": 90.0,
            "mean_interval_width": 10.0,
            "median_interval_width": 9.0,
            "low_risk_count": 12,
            "medium_risk_count": 8,
            "high_risk_count": 4,
        }
        return {
            "total_prediction_rows": 24,
            "total_metric_rows": 1,
            "latest_decision_run_id": "decision-run",
            "latest_created_at": datetime(2026, 1, 2, tzinfo=ISTANBUL),
            "available_model_versions": ["forecast_decision_v1"],
            "latest_selected_model": SELECTED_MODEL_XGBOOST,
            "latest_selection_reason": "test reason",
            "latest_metrics": metric,
            "latest_uncertainty_metrics": uncertainty,
            "latest_xgboost_comparison": {"selected_vs_xgboost_improvement_pct": 0.0},
            "latest_gpr_comparison": {"selected_vs_gpr_improvement_pct": 5.0},
        }


def test_forecast_decision_routes_are_registered_and_status_works() -> None:
    app.dependency_overrides[get_forecast_decision_ptf_service] = (
        lambda: FakeForecastDecisionStatusService()
    )
    try:
        client = TestClient(app)
        response = client.get("/api/models/forecast-decision/ptf/status")
        paths = app.openapi()["paths"]
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["latest_decision_run_id"] == "decision-run"
    assert "get" in paths["/api/models/forecast-decision/ptf/status"]
    assert "post" in paths["/api/models/forecast-decision/ptf/run"]
