import logging
import uuid
from collections.abc import Callable
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import mlflow
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal, engine
from ml.evaluation.metrics import calculate_regression_metrics

logger = logging.getLogger(__name__)

ISTANBUL_TIMEZONE = ZoneInfo("Europe/Istanbul")
DEFAULT_EVALUATION_START = date(2024, 1, 1)
MLFLOW_EXPERIMENT_NAME = "ptf_baseline_forecasting"
BASELINE_FEATURES = {
    "naive_lag_24": "ptf_lag_24",
    "seasonal_naive_lag_168": "ptf_lag_168",
    "rolling_24h_mean": "ptf_24h_mean",
    "rolling_7d_mean": "ptf_7d_mean",
}
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


class BaselinePtfService:
    def __init__(
        self,
        session_factory: Callable[[], Session] = SessionLocal,
        mlflow_tracking_uri: str | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.mlflow_tracking_uri = (
            mlflow_tracking_uri or settings.mlflow_tracking_uri
        )

    def load_feature_data(
        self,
        start_date: date | datetime | None = None,
        end_date: date | datetime | None = None,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        parameters: dict[str, datetime] = {}
        if start_date is not None:
            clauses.append('"timestamp" >= :start_timestamp')
            parameters["start_timestamp"] = _normalize_boundary(
                start_date,
                is_end=False,
            )
        if end_date is not None:
            clauses.append('"timestamp" <= :end_timestamp')
            parameters["end_timestamp"] = _normalize_boundary(
                end_date,
                is_end=True,
            )

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        feature_columns = ", ".join(
            f'"{column}"' for column in BASELINE_FEATURES.values()
        )
        query = text(
            f"""
            SELECT "timestamp", target_ptf, {feature_columns}
            FROM features_ptf_hourly
            {where_clause}
            ORDER BY "timestamp"
            """
        )
        with engine.connect() as connection:
            dataframe = pd.read_sql_query(query, connection, params=parameters)

        if dataframe.empty:
            return dataframe
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True)
        numeric_columns = ["target_ptf", *BASELINE_FEATURES.values()]
        for column in numeric_columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
        return dataframe.sort_values("timestamp").reset_index(drop=True)

    def generate_baseline_predictions(
        self,
        dataframe: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        required = {"timestamp", "target_ptf", *BASELINE_FEATURES.values()}
        missing = required.difference(dataframe.columns)
        if missing:
            raise ValueError(
                f"Missing baseline feature columns: {', '.join(sorted(missing))}"
            )

        predictions: dict[str, pd.DataFrame] = {}
        for model_name, feature_column in BASELINE_FEATURES.items():
            model_frame = dataframe[
                ["timestamp", "target_ptf", feature_column]
            ].copy()
            model_frame.columns = ["timestamp", "actual", "prediction"]
            model_frame["actual"] = pd.to_numeric(
                model_frame["actual"],
                errors="coerce",
            )
            model_frame["prediction"] = pd.to_numeric(
                model_frame["prediction"],
                errors="coerce",
            )
            finite = np.isfinite(model_frame["actual"]) & np.isfinite(
                model_frame["prediction"]
            )
            model_frame = model_frame.loc[finite].copy().reset_index(drop=True)
            model_frame["error"] = (
                model_frame["actual"] - model_frame["prediction"]
            )
            model_frame["absolute_error"] = model_frame["error"].abs()
            denominator = model_frame["actual"].abs().replace(0, np.nan)
            model_frame["percentage_error"] = (
                model_frame["absolute_error"] / denominator * 100
            )
            predictions[model_name] = model_frame
        return predictions

    def evaluate_baselines(
        self,
        dataframe: pd.DataFrame,
    ) -> dict[str, dict[str, float | int | None]]:
        predictions = self.generate_baseline_predictions(dataframe)
        return self._evaluate_prediction_frames(predictions)

    def store_predictions(
        self,
        predictions: dict[str, pd.DataFrame],
        evaluation_run_id: str,
        session: Session | None = None,
        batch_size: int = 2000,
    ) -> int:
        statement = text(
            """
            INSERT INTO baseline_predictions (
                "timestamp",
                model_name,
                prediction,
                actual,
                error,
                absolute_error,
                percentage_error,
                evaluation_run_id
            )
            VALUES (
                :timestamp,
                :model_name,
                :prediction,
                :actual,
                :error,
                :absolute_error,
                :percentage_error,
                :evaluation_run_id
            )
            ON CONFLICT ("timestamp", model_name, evaluation_run_id)
            DO UPDATE SET
                prediction = EXCLUDED.prediction,
                actual = EXCLUDED.actual,
                error = EXCLUDED.error,
                absolute_error = EXCLUDED.absolute_error,
                percentage_error = EXCLUDED.percentage_error
            """
        )
        rows: list[dict[str, Any]] = []
        for model_name, dataframe in predictions.items():
            for record in dataframe.to_dict(orient="records"):
                rows.append(
                    {
                        "timestamp": _python_value(record["timestamp"]),
                        "model_name": model_name,
                        "prediction": _python_value(record["prediction"]),
                        "actual": _python_value(record["actual"]),
                        "error": _python_value(record["error"]),
                        "absolute_error": _python_value(
                            record["absolute_error"]
                        ),
                        "percentage_error": _python_value(
                            record["percentage_error"]
                        ),
                        "evaluation_run_id": evaluation_run_id,
                    }
                )
        return self._execute_batches(statement, rows, session, batch_size)

    def store_metrics(
        self,
        metrics: dict[str, dict[str, float | int | None]],
        evaluation_run_id: str,
        start_timestamp: datetime | None,
        end_timestamp: datetime | None,
        session: Session | None = None,
    ) -> int:
        statement = text(
            """
            INSERT INTO baseline_metrics (
                evaluation_run_id,
                model_name,
                start_timestamp,
                end_timestamp,
                mae,
                rmse,
                mape,
                smape,
                r2,
                count,
                mean_actual,
                mean_prediction,
                max_error,
                median_absolute_error
            )
            VALUES (
                :evaluation_run_id,
                :model_name,
                :start_timestamp,
                :end_timestamp,
                :mae,
                :rmse,
                :mape,
                :smape,
                :r2,
                :count,
                :mean_actual,
                :mean_prediction,
                :max_error,
                :median_absolute_error
            )
            ON CONFLICT (evaluation_run_id, model_name)
            DO UPDATE SET
                start_timestamp = EXCLUDED.start_timestamp,
                end_timestamp = EXCLUDED.end_timestamp,
                mae = EXCLUDED.mae,
                rmse = EXCLUDED.rmse,
                mape = EXCLUDED.mape,
                smape = EXCLUDED.smape,
                r2 = EXCLUDED.r2,
                count = EXCLUDED.count,
                mean_actual = EXCLUDED.mean_actual,
                mean_prediction = EXCLUDED.mean_prediction,
                max_error = EXCLUDED.max_error,
                median_absolute_error = EXCLUDED.median_absolute_error
            """
        )
        rows = [
            {
                "evaluation_run_id": evaluation_run_id,
                "model_name": model_name,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                **metric_values,
            }
            for model_name, metric_values in metrics.items()
        ]
        return self._execute_batches(statement, rows, session, batch_size=100)

    def run_baseline_evaluation(
        self,
        start_date: date | datetime | None = None,
        end_date: date | datetime | None = None,
    ) -> dict[str, Any]:
        resolved_start = start_date or DEFAULT_EVALUATION_START
        if end_date is not None and _normalize_boundary(
            end_date,
            is_end=True,
        ) < _normalize_boundary(resolved_start, is_end=False):
            raise ValueError("end_date must be on or after start_date")

        evaluation_run_id = str(uuid.uuid4())
        dataframe = self.load_feature_data(resolved_start, end_date)
        predictions = self.generate_baseline_predictions(dataframe)
        metrics = self._evaluate_prediction_frames(predictions)
        warnings = [
            f"{model_name} has no valid rows in the selected evaluation range"
            for model_name, model_frame in predictions.items()
            if model_frame.empty
        ]
        errors: list[str] = []

        start_timestamp = (
            dataframe["timestamp"].min().to_pydatetime()
            if not dataframe.empty
            else None
        )
        end_timestamp = (
            dataframe["timestamp"].max().to_pydatetime()
            if not dataframe.empty
            else None
        )
        valid_predictions = {
            name: frame for name, frame in predictions.items() if not frame.empty
        }
        valid_metrics = {
            name: values
            for name, values in metrics.items()
            if values["count"] > 0
        }

        if not dataframe.empty and valid_predictions:
            with self.session_factory() as session:
                try:
                    self.store_predictions(
                        valid_predictions,
                        evaluation_run_id,
                        session=session,
                    )
                    self.store_metrics(
                        valid_metrics,
                        evaluation_run_id,
                        start_timestamp,
                        end_timestamp,
                        session=session,
                    )
                    session.commit()
                except SQLAlchemyError as exc:
                    session.rollback()
                    logger.exception("Could not persist baseline evaluation.")
                    errors.append(f"Database persistence failed: {exc}")
        else:
            warnings.append("No feature rows were available for baseline evaluation")

        if not errors and valid_metrics:
            mlflow_warning = self._log_to_mlflow(
                evaluation_run_id=evaluation_run_id,
                start_date=_date_label(resolved_start),
                end_date=(
                    _date_label(end_date)
                    if end_date is not None
                    else (
                        end_timestamp.astimezone(ISTANBUL_TIMEZONE)
                        .date()
                        .isoformat()
                        if end_timestamp is not None
                        else None
                    )
                ),
                metrics=valid_metrics,
            )
            if mlflow_warning:
                warnings.append(mlflow_warning)

        return {
            "evaluation_run_id": evaluation_run_id,
            "start_date": _date_label(resolved_start),
            "end_date": (
                _date_label(end_date)
                if end_date is not None
                else (
                    end_timestamp.astimezone(ISTANBUL_TIMEZONE)
                    .date()
                    .isoformat()
                    if end_timestamp is not None
                    else None
                )
            ),
            "models_evaluated": list(valid_metrics),
            "metrics": valid_metrics,
            "warnings": warnings,
            "errors": errors,
        }

    def get_status(self) -> dict[str, Any]:
        with self.session_factory() as session:
            counts = session.execute(
                text(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM baseline_predictions)
                            AS total_prediction_rows,
                        (SELECT COUNT(*) FROM baseline_metrics)
                            AS total_metric_rows
                    """
                )
            ).mappings().one()
            latest = session.execute(
                text(
                    """
                    SELECT evaluation_run_id, created_at
                    FROM baseline_metrics
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                )
            ).mappings().one_or_none()
            available_models = list(
                session.scalars(
                    text(
                        """
                        SELECT DISTINCT model_name
                        FROM baseline_metrics
                        ORDER BY model_name
                        """
                    )
                )
            )
            latest_metrics: dict[str, dict[str, Any]] = {}
            if latest is not None:
                metric_rows = session.execute(
                    text(
                        """
                        SELECT
                            model_name,
                            mae,
                            rmse,
                            mape,
                            smape,
                            r2,
                            count,
                            mean_actual,
                            mean_prediction,
                            max_error,
                            median_absolute_error
                        FROM baseline_metrics
                        WHERE evaluation_run_id = :evaluation_run_id
                        ORDER BY model_name
                        """
                    ),
                    {"evaluation_run_id": latest["evaluation_run_id"]},
                ).mappings()
                latest_metrics = {
                    row["model_name"]: {
                        key: _json_number(value)
                        for key, value in row.items()
                        if key != "model_name"
                    }
                    for row in metric_rows
                }

            return {
                **dict(counts),
                "latest_evaluation_run_id": (
                    latest["evaluation_run_id"] if latest else None
                ),
                "latest_created_at": latest["created_at"] if latest else None,
                "available_models": available_models,
                "latest_metrics": latest_metrics,
            }

    def _evaluate_prediction_frames(
        self,
        predictions: dict[str, pd.DataFrame],
    ) -> dict[str, dict[str, float | int | None]]:
        return {
            model_name: calculate_regression_metrics(
                model_frame["actual"],
                model_frame["prediction"],
            )
            for model_name, model_frame in predictions.items()
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

    def _log_to_mlflow(
        self,
        evaluation_run_id: str,
        start_date: str,
        end_date: str | None,
        metrics: dict[str, dict[str, float | int | None]],
    ) -> str | None:
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(
                run_name=f"baseline-{evaluation_run_id[:8]}",
                tags={"evaluation_run_id": evaluation_run_id},
            ):
                mlflow.log_params(
                    {
                        "start_date": start_date,
                        "end_date": end_date or "",
                        "model_names": ",".join(metrics),
                    }
                )
                for model_name, metric_values in metrics.items():
                    for metric_name, metric_value in metric_values.items():
                        if metric_value is not None:
                            mlflow.log_metric(
                                f"{model_name}_{metric_name}",
                                float(metric_value),
                            )
            return None
        except Exception as exc:
            logger.warning("MLflow baseline logging failed: %s", exc)
            return f"MLflow logging failed: {exc}"


def _normalize_boundary(value: date | datetime, is_end: bool) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=ISTANBUL_TIMEZONE)
        return value.astimezone(ISTANBUL_TIMEZONE)
    boundary = time(23, 59, 59) if is_end else time.min
    return datetime.combine(value, boundary, tzinfo=ISTANBUL_TIMEZONE)


def _date_label(value: date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _normalize_boundary(value, is_end=False).date().isoformat()
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
    if isinstance(value, Decimal):
        return float(value)
    return value
