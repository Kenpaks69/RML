"""
Chronological Walk-Forward Cross-Validation Engine.
Evaluates time-series forecasting models using strictly expanding historical windows.
"""

from typing import List, Dict, Any
import numpy as np

from .regression_metrics import compute_metrics
from ..forecasting.predict import DailyDemandForecaster, FEATURE_KEYS


class HistoricalMeanBaseline:
    def fit(self, y_train):
        self.mean_val = float(np.mean(y_train)) if len(y_train) > 0 else 35.0
        return self

    def predict(self, X_test):
        return np.full(len(X_test), self.mean_val)


class PreviousDayBaseline:
    def __init__(self, lag1_col_idx: int = 9):
        self.lag1_col_idx = lag1_col_idx

    def predict(self, X_test):
        return np.maximum(0.0, X_test[:, self.lag1_col_idx])


class RollingAverageBaseline:
    def __init__(self, rolling5_col_idx: int = 14):
        self.rolling5_col_idx = rolling5_col_idx

    def predict(self, X_test):
        return np.maximum(0.0, X_test[:, self.rolling5_col_idx])


def chronological_walk_forward_evaluation(
    daily_features: List[Dict[str, Any]],
    n_splits: int = 4,
    test_size: int = 25
) -> Dict[str, Any]:
    """
    Executes expanding-window chronological walk-forward cross-validation.
    Benchmarks candidate models: Random Forest, Decision Tree, Linear Regression, and Baselines.
    """
    total_records = len(daily_features)
    min_train_size = total_records - (n_splits * test_size)
    if min_train_size < 50:
        test_size = max(10, (total_records - 50) // n_splits)
        min_train_size = total_records - (n_splits * test_size)

    results = {
        "historical_mean": [],
        "previous_day": [],
        "rolling_5d_mean": [],
        "linear_regression": [],
        "decision_tree": [],
        "random_forest": []
    }

    split_details = []

    for split_idx in range(n_splits):
        train_end = min_train_size + (split_idx * test_size)
        test_end = train_end + test_size

        train_data = daily_features[:train_end]
        test_data = daily_features[train_end:test_end]

        X_train = np.array([[r[k] for k in FEATURE_KEYS] for r in train_data])
        y_train = np.array([r["target_requests"] for r in train_data])

        X_test = np.array([[r[k] for k in FEATURE_KEYS] for r in test_data])
        y_test = np.array([r["target_requests"] for r in test_data])

        # 1. Historical Mean
        b_mean = HistoricalMeanBaseline().fit(y_train)
        results["historical_mean"].append(compute_metrics(y_test, b_mean.predict(X_test)))

        # 2. Previous Day (lag_1 is index 9)
        b_prev = PreviousDayBaseline(lag1_col_idx=9)
        results["previous_day"].append(compute_metrics(y_test, b_prev.predict(X_test)))

        # 3. Rolling 5d Mean (rolling_mean_5 is index 14)
        b_roll = RollingAverageBaseline(rolling5_col_idx=14)
        results["rolling_5d_mean"].append(compute_metrics(y_test, b_roll.predict(X_test)))

        # 4. Linear Regression
        m_lr = DailyDemandForecaster(model_type="linear_regression").fit(X_train, y_train, FEATURE_KEYS)
        results["linear_regression"].append(compute_metrics(y_test, m_lr.predict(X_test)))

        # 5. Decision Tree Regressor
        m_dt = DailyDemandForecaster(model_type="decision_tree").fit(X_train, y_train, FEATURE_KEYS)
        results["decision_tree"].append(compute_metrics(y_test, m_dt.predict(X_test)))

        # 6. Random Forest Regressor (Champion)
        m_rf = DailyDemandForecaster(model_type="random_forest").fit(X_train, y_train, FEATURE_KEYS)
        results["random_forest"].append(compute_metrics(y_test, m_rf.predict(X_test)))

        split_details.append({
            "split": split_idx + 1,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "test_window": f"{test_data[0]['date']} to {test_data[-1]['date']}"
        })

    summary = {}
    for model_name, metric_list in results.items():
        summary[model_name] = {
            "mean_mae": round(float(np.mean([m["mae"] for m in metric_list])), 2),
            "mean_medae": round(float(np.mean([m["medae"] for m in metric_list])), 2),
            "mean_rmse": round(float(np.mean([m["rmse"] for m in metric_list])), 2),
            "mean_wape_pct": round(float(np.mean([m["wape_pct"] for m in metric_list])), 2)
        }

    return {
        "evaluation_type": "chronological_walk_forward",
        "n_splits": n_splits,
        "test_size_per_split": test_size,
        "split_windows": split_details,
        "model_comparison": summary,
        "models": summary,
        "detailed_splits": results
    }


def summarize_metrics(model_comparison: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardizes summary metrics dictionary with uniform keys (mae_mean, etc).
    """
    res = {}
    for m, vals in model_comparison.items():
        res[m] = {
            "mae_mean": vals.get("mean_mae", vals.get("mae_mean", 0.0)),
            "medae_mean": vals.get("mean_medae", vals.get("medae_mean", 0.0)),
            "rmse_mean": vals.get("mean_rmse", vals.get("rmse_mean", 0.0)),
            "wape_mean": vals.get("mean_wape_pct", vals.get("wape_mean", 0.0))
        }
    return res


evaluate_walk_forward_cv = chronological_walk_forward_evaluation

