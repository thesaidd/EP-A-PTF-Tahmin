import json
import logging
import uuid
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.session import SessionLocal, engine
from ml.evaluation.metrics import calculate_regression_metrics

logger = logging.getLogger(__name__)

DEFAULT_MODEL_VERSION = "forecast_decision_v1"
SELECTED_MODEL_XGBOOST = "xgboost"
SELECTED_MODEL_GPR = "gpr_corrected"
XGBOOST_SELECTION_REASON = (
    "GPR correction did not improve MAE over XGBoost on same window; using "
    "XGBoost as point forecast and GPR for uncertainty."
)
GPR_SELECTION_REASON = (
    "GPR correction improved MAE over XGBoost on same window; using "
    "GPR-corrected point forecast with GPR uncertainty."
)
METRIC_NAMES = (
    "mae",
    "rmse",
    "mape",
    "smape",
    "r2",
    "count",
    "mean_actual",
    "mean_prediction",
    "max_error",
    "median_absolute_error",
)
UNCERTAINTY_METRIC_NAMES = (
    "interval_coverage_95",
    "mean_interval_width",
    "median_interval_width",
    "low_risk_count",
    "medium_risk_count",
    "high_risk_count",
)


class ForecastDecisionPtfService:
    def __init__(
        self,
        session_factory: Callable[[], Session] = SessionLocal,
    ) -> None:
        self.session_factory = session_factory

    def get_latest_successful_gpr_run(self) -> str | None:
        with self.session_factory() as session:
            return session.scalar(
                text(
                    """
                    SELECT gpr_run_id
                    FROM gpr_residual_metrics
                    WHERE artifact_path IS NOT NULL
                      AND artifact_path <> ''
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                )
            )

    def load_gpr_metrics(self, gpr_run_id: str) -> dict[str, Any] | None:
        with self.session_factory() as session:
            row = session.execute(
                text(
                    """
                    SELECT *
                    FROM gpr_residual_metrics
                    WHERE gpr_run_id = :gpr_run_id
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                ),
                {"gpr_run_id": gpr_run_id},
            ).mappings().one_or_none()
        return dict(row) if row is not None else None

    def load_gpr_predictions(self, gpr_run_id: str) -> pd.DataFrame:
        query = text(
            """
            SELECT
                "timestamp",
                gpr_run_id,
                xgboost_training_run_id,
                xgboost_prediction,
                final_prediction AS gpr_corrected_prediction,
                actual,
                residual_mean,
                residual_std,
                risk_level
            FROM gpr_residual_predictions
            WHERE gpr_run_id = :gpr_run_id
            ORDER BY "timestamp"
            """
        )
        with engine.connect() as connection:
            dataframe = pd.read_sql_query(
                query,
                connection,
                params={"gpr_run_id": gpr_run_id},
            )
        if dataframe.empty:
            return dataframe
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True)
        numeric_columns = [
            "xgboost_prediction",
            "gpr_corrected_prediction",
            "actual",
            "residual_mean",
            "residual_std",
        ]
        for column in numeric_columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
        return dataframe.sort_values("timestamp").reset_index(drop=True)

    def decide_selected_model(
        self,
        gpr_metrics: dict[str, Any],
    ) -> tuple[str, str]:
        xgboost_comparison = gpr_metrics.get("xgboost_comparison") or {}
        improvement = _json_number(xgboost_comparison.get("mae_improvement_pct"))
        if improvement is not None and float(improvement) > 0:
            return SELECTED_MODEL_GPR, GPR_SELECTION_REASON
        return SELECTED_MODEL_XGBOOST, XGBOOST_SELECTION_REASON

    def build_decision_predictions(
        self,
        dataframe: pd.DataFrame,
        selected_model: str,
    ) -> pd.DataFrame:
        if selected_model not in {SELECTED_MODEL_XGBOOST, SELECTED_MODEL_GPR}:
            raise ValueError(f"Unsupported selected_model: {selected_model}")
        required = {
            "timestamp",
            "xgboost_prediction",
            "gpr_corrected_prediction",
            "actual",
            "residual_std",
            "risk_level",
        }
        missing = required.difference(dataframe.columns)
        if missing:
            raise ValueError(
                f"Missing decision input columns: {', '.join(sorted(missing))}"
            )

        frame = dataframe.copy().sort_values("timestamp").reset_index(drop=True)
        frame["selected_model"] = selected_model
        source_column = (
            "gpr_corrected_prediction"
            if selected_model == SELECTED_MODEL_GPR
            else "xgboost_prediction"
        )
        frame["selected_prediction"] = frame[source_column].astype(float)
        frame["lower_bound_95"] = (
            frame["selected_prediction"] - 1.96 * frame["residual_std"].astype(float)
        )
        frame["upper_bound_95"] = (
            frame["selected_prediction"] + 1.96 * frame["residual_std"].astype(float)
        )
        frame["interval_width_95"] = (
            frame["upper_bound_95"] - frame["lower_bound_95"]
        )
        frame["error"] = frame["actual"] - frame["selected_prediction"]
        frame["absolute_error"] = frame["error"].abs()
        denominator = frame["actual"].abs().replace(0, np.nan)
        frame["percentage_error"] = frame["absolute_error"] / denominator * 100
        finite_rows = (
            np.isfinite(frame["selected_prediction"])
            & np.isfinite(frame["actual"])
            & np.isfinite(frame["residual_std"])
        )
        return frame.loc[finite_rows].reset_index(drop=True)

    def calculate_decision_metrics(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        metrics = calculate_regression_metrics(
            dataframe["actual"],
            dataframe["selected_prediction"],
        )
        if dataframe.empty:
            uncertainty_metrics = {
                "interval_coverage_95": None,
                "mean_interval_width": None,
                "median_interval_width": None,
                "low_risk_count": 0,
                "medium_risk_count": 0,
                "high_risk_count": 0,
            }
        else:
            covered = (
                (dataframe["actual"] >= dataframe["lower_bound_95"])
                & (dataframe["actual"] <= dataframe["upper_bound_95"])
            )
            risk_counts = dataframe["risk_level"].value_counts()
            uncertainty_metrics = {
                "interval_coverage_95": float(covered.mean() * 100),
                "mean_interval_width": float(dataframe["interval_width_95"].mean()),
                "median_interval_width": float(dataframe["interval_width_95"].median()),
                "low_risk_count": int(risk_counts.get("LOW", 0)),
                "medium_risk_count": int(risk_counts.get("MEDIUM", 0)),
                "high_risk_count": int(risk_counts.get("HIGH", 0)),
            }
        return metrics, uncertainty_metrics

    def build_model_comparisons(
        self,
        dataframe: pd.DataFrame,
        selected_mae: float | int | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        xgboost_metrics = calculate_regression_metrics(
            dataframe["actual"],
            dataframe["xgboost_prediction"],
        )
        gpr_metrics = calculate_regression_metrics(
            dataframe["actual"],
            dataframe["gpr_corrected_prediction"],
        )
        xgboost_mae = xgboost_metrics["mae"]
        gpr_mae = gpr_metrics["mae"]
        return (
            {
                "xgboost_mae": xgboost_mae,
                "selected_mae": selected_mae,
                "selected_vs_xgboost_improvement_pct": _improvement_pct(
                    xgboost_mae,
                    selected_mae,
                ),
            },
            {
                "gpr_corrected_mae": gpr_mae,
                "selected_mae": selected_mae,
                "selected_vs_gpr_improvement_pct": _improvement_pct(
                    gpr_mae,
                    selected_mae,
                ),
            },
        )

    def store_decision_predictions(
        self,
        dataframe: pd.DataFrame,
        decision_run_id: str,
        gpr_run_id: str,
        xgboost_training_run_id: str,
        model_version: str,
        session: Session | None = None,
        batch_size: int = 2000,
    ) -> int:
        statement = text(
            """
            INSERT INTO forecast_decision_predictions (
                "timestamp",
                decision_run_id,
                gpr_run_id,
                xgboost_training_run_id,
                model_version,
                selected_model,
                xgboost_prediction,
                gpr_corrected_prediction,
                selected_prediction,
                actual,
                residual_mean,
                residual_std,
                lower_bound_95,
                upper_bound_95,
                interval_width_95,
                risk_level,
                error,
                absolute_error,
                percentage_error
            )
            VALUES (
                :timestamp,
                :decision_run_id,
                :gpr_run_id,
                :xgboost_training_run_id,
                :model_version,
                :selected_model,
                :xgboost_prediction,
                :gpr_corrected_prediction,
                :selected_prediction,
                :actual,
                :residual_mean,
                :residual_std,
                :lower_bound_95,
                :upper_bound_95,
                :interval_width_95,
                :risk_level,
                :error,
                :absolute_error,
                :percentage_error
            )
            ON CONFLICT ("timestamp", decision_run_id)
            DO UPDATE SET
                selected_model = EXCLUDED.selected_model,
                xgboost_prediction = EXCLUDED.xgboost_prediction,
                gpr_corrected_prediction = EXCLUDED.gpr_corrected_prediction,
                selected_prediction = EXCLUDED.selected_prediction,
                actual = EXCLUDED.actual,
                residual_mean = EXCLUDED.residual_mean,
                residual_std = EXCLUDED.residual_std,
                lower_bound_95 = EXCLUDED.lower_bound_95,
                upper_bound_95 = EXCLUDED.upper_bound_95,
                interval_width_95 = EXCLUDED.interval_width_95,
                risk_level = EXCLUDED.risk_level,
                error = EXCLUDED.error,
                absolute_error = EXCLUDED.absolute_error,
                percentage_error = EXCLUDED.percentage_error
            """
        )
        rows = [
            {
                "timestamp": _python_value(row["timestamp"]),
                "decision_run_id": decision_run_id,
                "gpr_run_id": gpr_run_id,
                "xgboost_training_run_id": xgboost_training_run_id,
                "model_version": model_version,
                "selected_model": row["selected_model"],
                "xgboost_prediction": _python_value(row["xgboost_prediction"]),
                "gpr_corrected_prediction": _python_value(
                    row["gpr_corrected_prediction"]
                ),
                "selected_prediction": _python_value(row["selected_prediction"]),
                "actual": _python_value(row["actual"]),
                "residual_mean": _python_value(row.get("residual_mean")),
                "residual_std": _python_value(row["residual_std"]),
                "lower_bound_95": _python_value(row["lower_bound_95"]),
                "upper_bound_95": _python_value(row["upper_bound_95"]),
                "interval_width_95": _python_value(row["interval_width_95"]),
                "risk_level": row["risk_level"],
                "error": _python_value(row["error"]),
                "absolute_error": _python_value(row["absolute_error"]),
                "percentage_error": _python_value(row["percentage_error"]),
            }
            for row in dataframe.to_dict(orient="records")
        ]
        return self._execute_batches(statement, rows, session, batch_size)

    def store_decision_metrics(
        self,
        decision_run_id: str,
        gpr_run_id: str,
        xgboost_training_run_id: str,
        model_version: str,
        selected_model: str,
        selection_reason: str,
        evaluation_start: datetime | None,
        evaluation_end: datetime | None,
        metrics: dict[str, Any],
        uncertainty_metrics: dict[str, Any],
        xgboost_comparison: dict[str, Any],
        gpr_comparison: dict[str, Any],
        decision_params: dict[str, Any],
        session: Session | None = None,
    ) -> int:
        statement = text(
            """
            INSERT INTO forecast_decision_metrics (
                decision_run_id,
                gpr_run_id,
                xgboost_training_run_id,
                model_version,
                selected_model,
                selection_reason,
                evaluation_start,
                evaluation_end,
                mae,
                rmse,
                mape,
                smape,
                r2,
                count,
                mean_actual,
                mean_prediction,
                max_error,
                median_absolute_error,
                interval_coverage_95,
                mean_interval_width,
                median_interval_width,
                low_risk_count,
                medium_risk_count,
                high_risk_count,
                xgboost_comparison,
                gpr_comparison,
                decision_params
            )
            VALUES (
                :decision_run_id,
                :gpr_run_id,
                :xgboost_training_run_id,
                :model_version,
                :selected_model,
                :selection_reason,
                :evaluation_start,
                :evaluation_end,
                :mae,
                :rmse,
                :mape,
                :smape,
                :r2,
                :count,
                :mean_actual,
                :mean_prediction,
                :max_error,
                :median_absolute_error,
                :interval_coverage_95,
                :mean_interval_width,
                :median_interval_width,
                :low_risk_count,
                :medium_risk_count,
                :high_risk_count,
                CAST(:xgboost_comparison AS JSONB),
                CAST(:gpr_comparison AS JSONB),
                CAST(:decision_params AS JSONB)
            )
            ON CONFLICT (decision_run_id) DO UPDATE SET
                selected_model = EXCLUDED.selected_model,
                selection_reason = EXCLUDED.selection_reason,
                xgboost_comparison = EXCLUDED.xgboost_comparison,
                gpr_comparison = EXCLUDED.gpr_comparison,
                decision_params = EXCLUDED.decision_params
            """
        )
        row = {
            "decision_run_id": decision_run_id,
            "gpr_run_id": gpr_run_id,
            "xgboost_training_run_id": xgboost_training_run_id,
            "model_version": model_version,
            "selected_model": selected_model,
            "selection_reason": selection_reason,
            "evaluation_start": evaluation_start,
            "evaluation_end": evaluation_end,
            **metrics,
            **uncertainty_metrics,
            "xgboost_comparison": json.dumps(_json_ready(xgboost_comparison)),
            "gpr_comparison": json.dumps(_json_ready(gpr_comparison)),
            "decision_params": json.dumps(_json_ready(decision_params)),
        }
        return self._execute_batches(statement, [row], session, batch_size=1)

    def run_decision_layer(
        self,
        gpr_run_id: str | None = None,
        model_version: str = DEFAULT_MODEL_VERSION,
    ) -> dict[str, Any]:
        if not model_version.strip():
            raise ValueError("model_version must not be empty")
        decision_run_id = str(uuid.uuid4())
        warnings: list[str] = []
        errors: list[str] = []
        resolved_gpr_run_id = gpr_run_id or self.get_latest_successful_gpr_run()
        if resolved_gpr_run_id is None:
            return self._summary(
                decision_run_id,
                "",
                "",
                model_version,
                "",
                "No successful GPR residual run was found.",
                None,
                None,
                {},
                {},
                {},
                {},
                warnings,
                ["No successful GPR residual run was found"],
            )
        gpr_metrics = self.load_gpr_metrics(resolved_gpr_run_id)
        if gpr_metrics is None:
            return self._summary(
                decision_run_id,
                resolved_gpr_run_id,
                "",
                model_version,
                "",
                "Requested GPR run was not found.",
                None,
                None,
                {},
                {},
                {},
                {},
                warnings,
                [f"GPR run not found: {resolved_gpr_run_id}"],
            )

        xgboost_training_run_id = str(gpr_metrics["xgboost_training_run_id"])
        selected_model, selection_reason = self.decide_selected_model(gpr_metrics)
        source_predictions = self.load_gpr_predictions(resolved_gpr_run_id)
        if source_predictions.empty:
            return self._summary(
                decision_run_id,
                resolved_gpr_run_id,
                xgboost_training_run_id,
                model_version,
                selected_model,
                selection_reason,
                None,
                None,
                {},
                {},
                {},
                {},
                warnings,
                [f"No GPR residual predictions found for run {resolved_gpr_run_id}"],
            )

        decision_predictions = self.build_decision_predictions(
            source_predictions,
            selected_model,
        )
        metrics, uncertainty_metrics = self.calculate_decision_metrics(
            decision_predictions
        )
        xgboost_comparison, gpr_comparison = self.build_model_comparisons(
            decision_predictions,
            metrics.get("mae"),
        )
        evaluation_start = _series_min_datetime(decision_predictions["timestamp"])
        evaluation_end = _series_max_datetime(decision_predictions["timestamp"])
        decision_params = {
            "selection_rule": "use_gpr_corrected_only_when_same_window_mae_improvement_pct_gt_0",
            "source_gpr_xgboost_comparison": _json_ready(
                gpr_metrics.get("xgboost_comparison") or {}
            ),
        }

        with self.session_factory() as session:
            try:
                self.store_decision_predictions(
                    decision_predictions,
                    decision_run_id,
                    resolved_gpr_run_id,
                    xgboost_training_run_id,
                    model_version,
                    session=session,
                )
                self.store_decision_metrics(
                    decision_run_id=decision_run_id,
                    gpr_run_id=resolved_gpr_run_id,
                    xgboost_training_run_id=xgboost_training_run_id,
                    model_version=model_version,
                    selected_model=selected_model,
                    selection_reason=selection_reason,
                    evaluation_start=evaluation_start,
                    evaluation_end=evaluation_end,
                    metrics=metrics,
                    uncertainty_metrics=uncertainty_metrics,
                    xgboost_comparison=xgboost_comparison,
                    gpr_comparison=gpr_comparison,
                    decision_params=decision_params,
                    session=session,
                )
                session.commit()
            except SQLAlchemyError as exc:
                session.rollback()
                logger.exception("Could not persist forecast decision results.")
                errors.append(f"Database persistence failed: {exc}")

        return self._summary(
            decision_run_id,
            resolved_gpr_run_id,
            xgboost_training_run_id,
            model_version,
            selected_model,
            selection_reason,
            evaluation_start,
            evaluation_end,
            metrics,
            uncertainty_metrics,
            xgboost_comparison,
            gpr_comparison,
            warnings,
            errors,
        )

    def get_status(self) -> dict[str, Any]:
        with self.session_factory() as session:
            counts = session.execute(
                text(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM forecast_decision_predictions)
                            AS total_prediction_rows,
                        (SELECT COUNT(*) FROM forecast_decision_metrics)
                            AS total_metric_rows
                    """
                )
            ).mappings().one()
            latest = session.execute(
                text(
                    """
                    SELECT *
                    FROM forecast_decision_metrics
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                )
            ).mappings().one_or_none()
            available_versions = list(
                session.scalars(
                    text(
                        """
                        SELECT DISTINCT model_version
                        FROM forecast_decision_metrics
                        ORDER BY model_version
                        """
                    )
                )
            )
        latest_metrics = None
        latest_uncertainty_metrics = None
        if latest is not None:
            latest_metrics = {
                metric_name: _json_number(latest[metric_name])
                for metric_name in METRIC_NAMES
            }
            latest_uncertainty_metrics = {
                metric_name: _json_number(latest[metric_name])
                for metric_name in UNCERTAINTY_METRIC_NAMES
            }
        return {
            **dict(counts),
            "latest_decision_run_id": latest["decision_run_id"] if latest else None,
            "latest_created_at": latest["created_at"] if latest else None,
            "available_model_versions": available_versions,
            "latest_selected_model": latest["selected_model"] if latest else None,
            "latest_selection_reason": latest["selection_reason"] if latest else None,
            "latest_metrics": latest_metrics,
            "latest_uncertainty_metrics": latest_uncertainty_metrics,
            "latest_xgboost_comparison": _json_ready(latest["xgboost_comparison"] or {})
            if latest
            else None,
            "latest_gpr_comparison": _json_ready(latest["gpr_comparison"] or {})
            if latest
            else None,
        }

    def _execute_batches(
        self,
        statement: Any,
        rows: list[dict[str, Any]],
        session: Session | None,
        batch_size: int,
    ) -> int:
        if not rows:
            return 0
        owns_session = session is None
        database_session = session or self.session_factory()
        affected_rows = 0
        try:
            for offset in range(0, len(rows), batch_size):
                batch = rows[offset : offset + batch_size]
                result = database_session.execute(statement, batch)
                affected_rows += (
                    result.rowcount if result.rowcount >= 0 else len(batch)
                )
            if owns_session:
                database_session.commit()
            return affected_rows
        except SQLAlchemyError:
            if owns_session:
                database_session.rollback()
            raise
        finally:
            if owns_session:
                database_session.close()

    def _summary(
        self,
        decision_run_id: str,
        gpr_run_id: str,
        xgboost_training_run_id: str,
        model_version: str,
        selected_model: str,
        selection_reason: str,
        evaluation_start: datetime | None,
        evaluation_end: datetime | None,
        metrics: dict[str, Any],
        uncertainty_metrics: dict[str, Any],
        xgboost_comparison: dict[str, Any],
        gpr_comparison: dict[str, Any],
        warnings: list[str],
        errors: list[str],
    ) -> dict[str, Any]:
        return {
            "decision_run_id": decision_run_id,
            "gpr_run_id": gpr_run_id,
            "xgboost_training_run_id": xgboost_training_run_id,
            "model_version": model_version,
            "selected_model": selected_model,
            "selection_reason": selection_reason,
            "evaluation_start": _datetime_label(evaluation_start),
            "evaluation_end": _datetime_label(evaluation_end),
            "metrics": _json_ready(metrics),
            "uncertainty_metrics": _json_ready(uncertainty_metrics),
            "xgboost_comparison": _json_ready(xgboost_comparison),
            "gpr_comparison": _json_ready(gpr_comparison),
            "warnings": warnings,
            "errors": errors,
        }


def _improvement_pct(
    reference_mae: float | int | None,
    selected_mae: float | int | None,
) -> float | None:
    if reference_mae in (None, 0) or selected_mae is None:
        return None
    return (float(reference_mae) - float(selected_mae)) / float(reference_mae) * 100


def _series_min_datetime(series: pd.Series) -> datetime | None:
    if series.empty:
        return None
    return _python_value(series.min())


def _series_max_datetime(series: pd.Series) -> datetime | None:
    if series.empty:
        return None
    return _python_value(series.max())


def _datetime_label(value: datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value.isoformat()


def _python_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value
