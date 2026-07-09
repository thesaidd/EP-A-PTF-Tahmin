from datetime import datetime
from math import sqrt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.models import get_baseline_ptf_service
from app.main import app
from ml.evaluation.metrics import (
    calculate_regression_metrics,
    mape_safe,
    mean_absolute_error_safe,
    r2_safe,
    rmse_safe,
    smape_safe,
)
from ml.models.baseline_ptf import BASELINE_FEATURES, BaselinePtfService

ISTANBUL = ZoneInfo("Europe/Istanbul")


def baseline_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=4,
                freq="h",
                tz=ISTANBUL,
            ),
            "target_ptf": [100.0, 120.0, 140.0, 160.0],
            "ptf_lag_24": [90.0, np.nan, 130.0, 150.0],
            "ptf_lag_168": [80.0, 100.0, 120.0, 140.0],
            "ptf_24h_mean": [95.0, 105.0, 125.0, 145.0],
            "ptf_7d_mean": [85.0, 105.0, 125.0, 145.0],
        }
    )


def test_safe_regression_metrics() -> None:
    actual = [1.0, 2.0, 3.0]
    prediction = [1.0, 2.0, 5.0]

    assert mean_absolute_error_safe(actual, prediction) == pytest.approx(2 / 3)
    assert rmse_safe(actual, prediction) == pytest.approx(sqrt(4 / 3))
    assert r2_safe(actual, prediction) == pytest.approx(-1.0)
    assert mape_safe(actual, prediction) == pytest.approx((2 / 3) / 3 * 100)
    assert smape_safe(actual, prediction) is not None

    metrics = calculate_regression_metrics(actual, prediction)
    assert metrics["count"] == 3
    assert metrics["max_error"] == 2.0
    assert metrics["median_absolute_error"] == 0.0


def test_mape_ignores_zero_actual_values() -> None:
    assert mape_safe([0.0, 10.0], [5.0, 8.0]) == pytest.approx(20.0)
    assert mape_safe([0.0], [5.0]) is None


def test_baseline_prediction_generation_drops_null_predictions() -> None:
    service = BaselinePtfService()

    predictions = service.generate_baseline_predictions(baseline_frame())

    assert set(predictions) == set(BASELINE_FEATURES)
    assert len(predictions["naive_lag_24"]) == 3
    assert len(predictions["seasonal_naive_lag_168"]) == 4
    assert predictions["naive_lag_24"]["prediction"].isna().sum() == 0
    assert predictions["naive_lag_24"].iloc[0]["error"] == 10.0


class FakeSession:
    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class InMemoryBaselineService(BaselinePtfService):
    def __init__(self) -> None:
        super().__init__(
            session_factory=FakeSession,  # type: ignore[arg-type]
            mlflow_tracking_uri="http://unused",
        )

    def load_feature_data(
        self,
        start_date: object = None,
        end_date: object = None,
    ) -> pd.DataFrame:
        return baseline_frame()

    def store_predictions(self, *args: object, **kwargs: object) -> int:
        return 15

    def store_metrics(self, *args: object, **kwargs: object) -> int:
        return 4

    def _log_to_mlflow(self, *args: object, **kwargs: object) -> str | None:
        return None


def test_evaluation_summary_shape_without_external_services() -> None:
    summary = InMemoryBaselineService().run_baseline_evaluation()

    assert summary["evaluation_run_id"]
    assert summary["start_date"] == "2024-01-01"
    assert set(summary["models_evaluated"]) == set(BASELINE_FEATURES)
    assert summary["metrics"]["naive_lag_24"]["count"] == 3
    assert summary["warnings"] == []
    assert summary["errors"] == []


class FakeStatusService:
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
        return {
            "total_prediction_rows": 96,
            "total_metric_rows": 4,
            "latest_evaluation_run_id": "test-run",
            "latest_created_at": datetime(2024, 1, 2, tzinfo=ISTANBUL),
            "available_models": list(BASELINE_FEATURES),
            "latest_metrics": {"naive_lag_24": metric},
        }


def test_baseline_routes_are_registered_and_status_works() -> None:
    app.dependency_overrides[get_baseline_ptf_service] = lambda: FakeStatusService()
    try:
        client = TestClient(app)
        response = client.get("/api/models/baseline/ptf/status")
        paths = app.openapi()["paths"]
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["latest_evaluation_run_id"] == "test-run"
    assert "get" in paths["/api/models/baseline/ptf/status"]
    assert "post" in paths["/api/models/baseline/ptf/run"]
