"""
Daily Demand Forecaster Training and Model Selection.
Trains Champion Random Forest Regressor and serializes artifact.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import joblib

from ..common.paths import (
    DAILY_FORECAST_MODELS_DIR,
    ARTIFACTS_DIR
)
from .predict import DailyDemandForecaster, FEATURE_KEYS


def train_daily_forecaster(
    daily_features: List[Dict[str, Any]],
    model_type: str = "random_forest",
    output_path: Optional[str] = None
) -> DailyDemandForecaster:
    """
    Trains the DailyDemandForecaster model on all feature observations.
    Serializes to target directory and legacy root artifacts directory.
    """
    X = np.array([[r[k] for k in FEATURE_KEYS] for r in daily_features])
    y = np.array([r["target_requests"] for r in daily_features])

    forecaster = DailyDemandForecaster(model_type=model_type)
    forecaster.fit(X, y, FEATURE_KEYS)

    # Save to models/daily_forecast/
    save_path = output_path or str(DAILY_FORECAST_MODELS_DIR / "daily_forecaster.joblib")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(forecaster, save_path)

    # Also save to legacy root artifacts/ for backwards compatibility
    legacy_path = str(ARTIFACTS_DIR / "daily_forecaster.joblib")
    joblib.dump(forecaster, legacy_path)

    return forecaster
