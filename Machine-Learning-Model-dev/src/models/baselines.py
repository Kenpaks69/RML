"""
Baseline Forecasting Models for Daily Incoming Requests.
Serves as benchmark models for chronological walk-forward evaluation.
"""

from typing import List, Dict, Any
import numpy as np


class HistoricalMeanBaseline:
    """Predicts the overall historical mean of training demand."""
    def __init__(self):
        self.mean_value = 0.0

    def fit(self, y: np.ndarray):
        self.mean_value = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(shape=(len(X),), fill_value=self.mean_value)


class PreviousDayBaseline:
    """Predicts the most recent operating day's count (lag 1 / persistence)."""
    def __init__(self, lag1_col_idx: int = 0):
        self.lag1_col_idx = lag1_col_idx

    def fit(self, y: np.ndarray):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.lag1_col_idx]


class PreviousSameWeekdayBaseline:
    """Predicts the count from the same weekday in the previous week (lag 5)."""
    def __init__(self, lag5_col_idx: int = 1):
        self.lag5_col_idx = lag5_col_idx

    def fit(self, y: np.ndarray):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.lag5_col_idx]


class RollingAverageBaseline:
    """Predicts the 5-day rolling average of recent operating days."""
    def __init__(self, rolling5_col_idx: int = 2):
        self.rolling5_col_idx = rolling5_col_idx

    def fit(self, y: np.ndarray):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.rolling5_col_idx]

