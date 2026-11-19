"""
Evaluation Metrics for Regression and Count Forecasting.
Calculates MAE, MedAE, RMSE, WAPE, and Poisson Deviance.
"""

import math
from typing import Dict, Any
import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates comprehensive forecasting error metrics."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.maximum(0.0, np.array(y_pred, dtype=float))

    n = len(y_true)
    if n == 0:
        return {"mae": 0.0, "medae": 0.0, "rmse": 0.0, "wape_pct": 0.0}

    abs_errors = np.abs(y_true - y_pred)
    mae = float(np.mean(abs_errors))
    medae = float(np.median(abs_errors))
    rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))

    total_actual = float(np.sum(y_true))
    wape = float(np.sum(abs_errors) / total_actual * 100.0) if total_actual > 0 else 0.0

    # Poisson deviance
    eps = 1e-6
    y_pred_safe = np.maximum(y_pred, eps)
    deviance_terms = []
    for yt, yp in zip(y_true, y_pred_safe):
        if yt > 0:
            deviance_terms.append(yt * math.log(yt / yp) - (yt - yp))
        else:
            deviance_terms.append(yp)
    poisson_deviance = float(2.0 * np.mean(deviance_terms))

    return {
        "mae": round(mae, 2),
        "medae": round(medae, 2),
        "rmse": round(rmse, 2),
        "wape_pct": round(wape, 2),
        "poisson_deviance": round(poisson_deviance, 2)
    }


calculate_regression_metrics = compute_metrics

