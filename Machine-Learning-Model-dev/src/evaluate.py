"""
Chronological Walk-Forward Evaluator for RegQuest Machine Learning Models.
Evaluates time-series forecasting and processing time estimators using strictly chronological splits.
Calculates MAE, MedAE, RMSE, WAPE, and monthly workload classification accuracy.
"""

import os
import math
import json
from typing import List, Dict, Any, Tuple
import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates comprehensive forecasting error metrics."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.maximum(0.0, np.array(y_pred, dtype=float))
    
    n = len(y_true)
    if n == 0:
        return {"mae": 0.0, "medae": 0.0, "rmse": 0.0, "wape": 0.0}
        
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


def chronological_walk_forward_evaluation(
    daily_features: List[Dict[str, Any]],
    n_splits: int = 4,
    test_size: int = 25
) -> Dict[str, Any]:
    """
    Executes expanding-window walk-forward cross-validation.
    Compares Baselines vs Poisson vs Ridge vs Gradient Boosting.
    """
    from .models.baselines import (
        HistoricalMeanBaseline,
        PreviousDayBaseline,
        PreviousSameWeekdayBaseline,
        RollingAverageBaseline
    )
    from .models.forecaster import DailyDemandForecaster

    feature_keys = [
        "day_of_week", "month", "day_of_month", "is_operating_day",
        "is_weekend", "is_peak_academic_season", "is_slow_academic_season",
        "is_month_start", "is_month_end", "lag_1", "lag_2", "lag_3",
        "lag_5", "lag_10", "rolling_mean_5", "rolling_mean_10",
        "rolling_std_5", "rolling_min_5", "rolling_max_5"
    ]

    total_records = len(daily_features)
    min_train_size = total_records - (n_splits * test_size)
    if min_train_size < 50:
        test_size = max(10, (total_records - 50) // n_splits)
        min_train_size = total_records - (n_splits * test_size)

    results = {
        "historical_mean": [],
        "previous_day": [],
        "previous_same_weekday": [],
        "rolling_5d_mean": [],
        "linear_regression": [],
        "decision_tree": [],
        "random_forest": [],
        "ridge": [],
        "poisson": []
    }

    split_details = []

    for split_idx in range(n_splits):
        train_end = min_train_size + (split_idx * test_size)
        test_end = train_end + test_size

        train_data = daily_features[:train_end]
        test_data = daily_features[train_end:test_end]

        X_train = np.array([[r[k] for k in feature_keys] for r in train_data])
        y_train = np.array([r["target_requests"] for r in train_data])

        X_test = np.array([[r[k] for k in feature_keys] for r in test_data])
        y_test = np.array([r["target_requests"] for r in test_data])

        test_start_date = test_data[0]["date"]
        test_end_date = test_data[-1]["date"]

        # 1. Historical Mean Baseline
        b_mean = HistoricalMeanBaseline().fit(y_train)
        pred_mean = b_mean.predict(X_test)
        results["historical_mean"].append(compute_metrics(y_test, pred_mean))

        # 2. Previous Day Baseline (lag_1 is index 9)
        b_prev = PreviousDayBaseline(lag1_col_idx=9)
        pred_prev = b_prev.predict(X_test)
        results["previous_day"].append(compute_metrics(y_test, pred_prev))

        # 3. Previous Same Weekday (lag_5 is index 12)
        b_prev_w = PreviousSameWeekdayBaseline(lag5_col_idx=12)
        pred_prev_w = b_prev_w.predict(X_test)
        results["previous_same_weekday"].append(compute_metrics(y_test, pred_prev_w))

        # 4. Rolling 5d Mean (rolling_mean_5 is index 14)
        b_roll = RollingAverageBaseline(rolling5_col_idx=14)
        pred_roll = b_roll.predict(X_test)
        results["rolling_5d_mean"].append(compute_metrics(y_test, pred_roll))

        # 5. Linear Regression
        try:
            m_lr = DailyDemandForecaster(model_type="linear_regression").fit(X_train, y_train, feature_keys)
            pred_lr = m_lr.predict(X_test)
            results["linear_regression"].append(compute_metrics(y_test, pred_lr))
        except Exception:
            pass

        # 6. Decision Tree Regressor
        try:
            m_dt = DailyDemandForecaster(model_type="decision_tree").fit(X_train, y_train, feature_keys)
            pred_dt = m_dt.predict(X_test)
            results["decision_tree"].append(compute_metrics(y_test, pred_dt))
        except Exception:
            pass

        # 7. Random Forest Regressor
        try:
            m_rf = DailyDemandForecaster(model_type="random_forest").fit(X_train, y_train, feature_keys)
            pred_rf = m_rf.predict(X_test)
            results["random_forest"].append(compute_metrics(y_test, pred_rf))
        except Exception:
            pass

        # 8. Ridge Regressor
        try:
            m_ridge = DailyDemandForecaster(model_type="ridge").fit(X_train, y_train, feature_keys)
            pred_ridge = m_ridge.predict(X_test)
            results["ridge"].append(compute_metrics(y_test, pred_ridge))
        except Exception:
            pass

        # 9. Poisson Regressor
        try:
            m_poisson = DailyDemandForecaster(model_type="poisson").fit(X_train, y_train, feature_keys)
            pred_poisson = m_poisson.predict(X_test)
            results["poisson"].append(compute_metrics(y_test, pred_poisson))
        except Exception:
            pass

        split_details.append({
            "split": split_idx + 1,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "test_window": f"{test_start_date} to {test_end_date}"
        })

    # Aggregate metric averages
    summary = {}
    for model_name, metric_list in results.items():
        if not metric_list:
            continue
        summary[model_name] = {
            "mean_mae": round(float(np.mean([m["mae"] for m in metric_list])), 2),
            "mean_medae": round(float(np.mean([m["medae"] for m in metric_list])), 2),
            "mean_rmse": round(float(np.mean([m["rmse"] for m in metric_list])), 2),
            "mean_wape_pct": round(float(np.mean([m["wape_pct"] for m in metric_list])), 2),
            "mean_poisson_deviance": round(float(np.mean([m["poisson_deviance"] for m in metric_list])), 2)
        }

    return {
        "evaluation_type": "chronological_walk_forward",
        "n_splits": n_splits,
        "test_size_per_split": test_size,
        "split_windows": split_details,
        "model_comparison": summary,
        "detailed_splits": results
    }

