"""
Daily Request Demand Forecaster and Autoregressive Rollout Engine.
Supports Champion Random Forest Regressor and baseline candidates.
"""

import math
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge


FEATURE_KEYS: List[str] = [
    "day_of_week", "month", "day_of_month", "is_operating_day",
    "is_weekend", "is_peak_academic_season", "is_slow_academic_season",
    "is_month_start", "is_month_end", "lag_1", "lag_2", "lag_3",
    "lag_5", "lag_10", "rolling_mean_5", "rolling_mean_10",
    "rolling_std_5", "rolling_min_5", "rolling_max_5"
]


class DailyDemandForecaster:
    """
    Forecasting pipeline wrapping trained regressor with autoregressive rollout
    and 90% confidence prediction intervals.
    """
    def __init__(self, model_type: str = "random_forest", alpha: float = 1.0):
        self.model_type = model_type
        self.alpha = alpha

        if model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
        elif model_type == "decision_tree":
            self.model = DecisionTreeRegressor(max_depth=3, random_state=42)
        elif model_type == "linear_regression":
            self.model = LinearRegression()
        elif model_type == "ridge":
            self.model = Ridge(alpha=alpha)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.feature_names = list(FEATURE_KEYS)
        self.residual_std = 10.0
        self.model_version = f"daily-forecaster-{model_type}-v1"

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or list(FEATURE_KEYS)
        self.model.fit(X, y)
        preds = np.maximum(0.0, self.model.predict(X))
        residuals = y - preds
        self.residual_std = float(np.std(residuals))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw_preds = self.model.predict(X)
        return np.maximum(0.0, raw_preds)

    def autoregressive_rollout(
        self,
        operating_dates: List[str],
        recent_history: List[float],
        peak_months: set = None
    ) -> List[Dict[str, Any]]:
        """
        Rolls forward multi-step daily forecasts autoregressively,
        updating lag and rolling window features at each step.
        """
        peak_months = peak_months or {1, 6, 8, 9}
        slow_months = {3, 12}

        history_buffer = list(recent_history)
        results = []

        for d_str in operating_dates:
            dt = datetime.strptime(d_str, "%Y-%m-%d")
            day_of_week = dt.weekday()
            month = dt.month
            day_of_month = dt.day

            is_peak = int(month in peak_months)
            is_slow = int(month in slow_months)
            is_month_start = int(day_of_month <= 5)
            is_month_end = int(day_of_month >= 25)

            lag_1 = history_buffer[-1] if len(history_buffer) >= 1 else 35.0
            lag_2 = history_buffer[-2] if len(history_buffer) >= 2 else lag_1
            lag_3 = history_buffer[-3] if len(history_buffer) >= 3 else lag_2
            lag_5 = history_buffer[-5] if len(history_buffer) >= 5 else lag_1
            lag_10 = history_buffer[-10] if len(history_buffer) >= 10 else lag_5

            w5 = history_buffer[-5:] if len(history_buffer) >= 5 else history_buffer
            w10 = history_buffer[-10:] if len(history_buffer) >= 10 else history_buffer

            mean_5 = sum(w5) / len(w5)
            mean_10 = sum(w10) / len(w10)
            var_5 = sum((x - mean_5) ** 2 for x in w5) / len(w5) if len(w5) > 1 else 0.0
            std_5 = math.sqrt(var_5)
            min_5 = min(w5)
            max_5 = max(w5)

            feat_vec = np.array([[
                day_of_week,
                month,
                day_of_month,
                1,  # is_operating_day
                0,  # is_weekend
                is_peak,
                is_slow,
                is_month_start,
                is_month_end,
                lag_1,
                lag_2,
                lag_3,
                lag_5,
                lag_10,
                mean_5,
                mean_10,
                std_5,
                min_5,
                max_5
            ]])

            pred_val = float(np.maximum(0.0, self.predict(feat_vec)[0]))
            lower_bound = max(0, int(round(pred_val - 1.645 * self.residual_std)))
            upper_bound = int(round(pred_val + 1.645 * self.residual_std))

            results.append({
                "date": d_str,
                "day_of_week": dt.strftime("%A"),
                "weekday": dt.strftime("%A"),
                "predicted_requests": int(round(pred_val)),
                "predicted_requests_raw": round(pred_val, 2),
                "lower_bound_90": lower_bound,
                "upper_bound_90": upper_bound,
                "is_operating_day": True
            })

            history_buffer.append(pred_val)

        return results
