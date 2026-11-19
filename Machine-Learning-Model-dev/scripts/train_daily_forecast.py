"""
Script: train_daily_forecast.py
Trains champion Random Forest Regressor for daily demand forecasting.
Serializes artifact to artifacts/models/daily_forecast/ and legacy artifacts/.
Writes metadata to artifacts/metadata/ and legacy artifacts/.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add ML root to path
ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from src.regquest_ml.common.paths import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    DAILY_FORECAST_MODELS_DIR,
    METADATA_DIR,
    ARTIFACTS_DIR,
)
from src.regquest_ml.data.loader import load_raw_csv_files, load_processed_data
from src.regquest_ml.data.cleaner import clean_raw_records
from src.regquest_ml.features.daily_demand import (
    aggregate_daily_time_series,
    build_daily_features_table,
)
from src.regquest_ml.forecasting.train import train_daily_forecaster
from src.regquest_ml.evaluation.regression_metrics import compute_metrics


def main():
    print("=" * 60)
    print("Starting Daily Demand Champion Model Training")
    print("=" * 60)

    # 1. Load data
    ts_path = PROCESSED_DATA_DIR / "daily_time_series.csv"
    if ts_path.exists():
        print(f"Loading daily time series from: {ts_path}")
        daily_series = load_processed_data(str(ts_path))
    else:
        processed_path = PROCESSED_DATA_DIR / "cleaned_requests.csv"
        if processed_path.exists():
            print(f"Loading cleaned records from: {processed_path}")
            records = load_processed_data(str(processed_path))
        else:
            print(f"Processing raw datasets from: {RAW_DATA_DIR}")
            raw_records, _ = load_raw_csv_files(str(RAW_DATA_DIR))
            cleaning_result = clean_raw_records(raw_records)
            records = cleaning_result["cleaned"]
        daily_series = aggregate_daily_time_series(records)

    print(f"Total operating days loaded: {len(daily_series)}")


    daily_features = build_daily_features_table(daily_series)
    print(f"Feature table observations built: {len(daily_features)}")

    # 3. Train Champion Random Forest Model
    print("Training Champion Random Forest Regressor (n_estimators=50, max_depth=4)...")
    forecaster = train_daily_forecaster(daily_features, model_type="random_forest")

    # In-sample performance check
    import numpy as np
    from src.regquest_ml.forecasting.predict import FEATURE_KEYS
    X = np.array([[r[k] for k in FEATURE_KEYS] for r in daily_features])
    y = np.array([r["target_requests"] for r in daily_features])
    preds = forecaster.predict(X)
    metrics = compute_metrics(y, preds)

    print("\nTraining Metrics (In-Sample Fit):")
    print(f"  - MAE:   {metrics['mae']} requests/day")
    print(f"  - MedAE: {metrics['medae']} requests/day")
    print(f"  - RMSE:  {metrics['rmse']} requests/day")
    print(f"  - Residual Std: {forecaster.residual_std:.2f}")

    # 4. Prepare and persist metadata
    recent_history = [float(r["target_requests"]) for r in daily_features[-15:]]
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "total_operating_days": len(daily_series),
        "recent_history_buffer": recent_history,
        "champion_model": {
            "name": "RandomForestRegressor",
            "model_type": "random_forest",
            "n_estimators": 50,
            "max_depth": 4,
            "version": forecaster.model_version,
            "residual_std": round(forecaster.residual_std, 2)
        },
        "daily_forecaster": {
            "version": forecaster.model_version,
            "feature_count": len(forecaster.feature_names)
        },
        "processing_time_baseline": {
            "version": "processing-baseline-v1"
        },
        "monthly_workload_thresholds": {
            "light_max": 500,
            "moderate_max": 850,
            "heavy_max": 1150
        }
    }

    meta_target = METADATA_DIR / "metadata.json"
    legacy_meta = ARTIFACTS_DIR / "metadata.json"
    os.makedirs(METADATA_DIR, exist_ok=True)

    with open(meta_target, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
    with open(legacy_meta, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"\nMetadata saved to:")
    print(f"  -> {meta_target}")
    print(f"  -> {legacy_meta}")
    print("=" * 60)


if __name__ == "__main__":
    main()
