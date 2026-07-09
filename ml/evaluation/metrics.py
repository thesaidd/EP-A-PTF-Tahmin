from typing import Any

import numpy as np
import pandas as pd


def mean_absolute_error_safe(
    actual: Any,
    prediction: Any,
) -> float | None:
    actual_values, prediction_values = _paired_values(actual, prediction)
    if actual_values.size == 0:
        return None
    return float(np.mean(np.abs(actual_values - prediction_values)))


def rmse_safe(actual: Any, prediction: Any) -> float | None:
    actual_values, prediction_values = _paired_values(actual, prediction)
    if actual_values.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(actual_values - prediction_values))))


def mape_safe(actual: Any, prediction: Any) -> float | None:
    actual_values, prediction_values = _paired_values(actual, prediction)
    nonzero = actual_values != 0
    if not np.any(nonzero):
        return None
    return float(
        np.mean(
            np.abs(
                (actual_values[nonzero] - prediction_values[nonzero])
                / actual_values[nonzero]
            )
        )
        * 100
    )


def smape_safe(actual: Any, prediction: Any) -> float | None:
    actual_values, prediction_values = _paired_values(actual, prediction)
    denominator = np.abs(actual_values) + np.abs(prediction_values)
    valid = denominator != 0
    if not np.any(valid):
        return None
    return float(
        np.mean(
            2
            * np.abs(actual_values[valid] - prediction_values[valid])
            / denominator[valid]
        )
        * 100
    )


def r2_safe(actual: Any, prediction: Any) -> float | None:
    actual_values, prediction_values = _paired_values(actual, prediction)
    if actual_values.size < 2:
        return None
    total_sum_squares = np.sum(np.square(actual_values - np.mean(actual_values)))
    if total_sum_squares == 0:
        return None
    residual_sum_squares = np.sum(np.square(actual_values - prediction_values))
    return float(1 - residual_sum_squares / total_sum_squares)


def calculate_regression_metrics(
    actual: Any,
    prediction: Any,
) -> dict[str, float | int | None]:
    actual_values, prediction_values = _paired_values(actual, prediction)
    if actual_values.size == 0:
        return {
            "mae": None,
            "rmse": None,
            "mape": None,
            "smape": None,
            "r2": None,
            "count": 0,
            "mean_actual": None,
            "mean_prediction": None,
            "max_error": None,
            "median_absolute_error": None,
        }

    absolute_errors = np.abs(actual_values - prediction_values)
    return {
        "mae": mean_absolute_error_safe(actual_values, prediction_values),
        "rmse": rmse_safe(actual_values, prediction_values),
        "mape": mape_safe(actual_values, prediction_values),
        "smape": smape_safe(actual_values, prediction_values),
        "r2": r2_safe(actual_values, prediction_values),
        "count": int(actual_values.size),
        "mean_actual": float(np.mean(actual_values)),
        "mean_prediction": float(np.mean(prediction_values)),
        "max_error": float(np.max(absolute_errors)),
        "median_absolute_error": float(np.median(absolute_errors)),
    }


def _paired_values(actual: Any, prediction: Any) -> tuple[np.ndarray, np.ndarray]:
    actual_values = pd.to_numeric(pd.Series(actual), errors="coerce").to_numpy(
        dtype=float
    )
    prediction_values = pd.to_numeric(
        pd.Series(prediction),
        errors="coerce",
    ).to_numpy(dtype=float)
    if actual_values.shape != prediction_values.shape:
        raise ValueError("actual and prediction must have the same length")
    valid = np.isfinite(actual_values) & np.isfinite(prediction_values)
    return actual_values[valid], prediction_values[valid]

