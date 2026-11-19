"""
Master Model Training, Chronological Validation, and Artifact Serialization Script.
Trains:
1. Daily Demand Forecaster (Walk-forward validated count regressor)
2. Processing Time Turnaround Baseline Estimator
3. Workload Forecasting and Classification Rules
Serializes versioned artifacts and metadata into ML/artifacts/.
"""

import os
import csv
import json
from datetime import datetime
import joblib
import numpy as np

from .pipeline import run_pipeline, PROCESSED_DATA_DIR
from .features import extract_daily_features, PEAK_ACADEMIC_MONTHS
from .models.forecaster import DailyDemandForecaster
from .models.processing_time import ProcessingTimeBaseline
from .evaluate import chronological_walk_forward_evaluation

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


def load_daily_ts() -> list:
    daily_csv = os.path.join(PROCESSED_DATA_DIR, "daily_time_series.csv")
    if not os.path.exists(daily_csv):
        run_pipeline()
    records = []
    with open(daily_csv, 'r', encoding='utf-8') as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            records.append({
                "date": r["date"],
                "total_requests": float(r["total_requests"]),
                "is_operating_day": int(r["is_operating_day"]),
                "is_weekend": int(r["is_weekend"])
            })
    return records


def load_cleaned_requests() -> list:
    req_csv = os.path.join(PROCESSED_DATA_DIR, "cleaned_requests.csv")
    if not os.path.exists(req_csv):
        run_pipeline()
    records = []
    with open(req_csv, 'r', encoding='utf-8') as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            biz_days = float(r["scheduled_lead_business_days"]) if r.get("scheduled_lead_business_days") else None
            records.append({
                "primary_credential": r["primary_credential"],
                "purpose_category": r["purpose_category"],
                "is_multi_document": r["is_multi_document"] == "True",
                "scheduled_lead_business_days": biz_days
            })
    return records


def train_and_persist():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print("=== RegQuest Model Training & Chronological Validation ===")

    # 1. Load data
    daily_records = load_daily_ts()
    cleaned_requests = load_cleaned_requests()
    print(f"Loaded {len(daily_records)} daily time-series records and {len(cleaned_requests)} request records.")

    # 2. Extract strictly backward-looking time-series features
    daily_features = extract_daily_features(daily_records)
    print(f"Computed {len(daily_features)} featured daily observations.")

    # 3. Chronological Walk-Forward Cross-Validation
    print("Executing walk-forward cross-validation (4 expanding splits)...")
    eval_results = chronological_walk_forward_evaluation(daily_features, n_splits=4, test_size=25)
    print("\n--- Model Performance Comparison (Mean Metrics across Chronological Splits) ---")
    for m_name, metrics in eval_results["model_comparison"].items():
        print(f"  {m_name:24s} | MAE: {metrics['mean_mae']:5.2f} | MedAE: {metrics['mean_medae']:5.2f} | RMSE: {metrics['mean_rmse']:5.2f} | WAPE: {metrics['mean_wape_pct']:5.2f}%")

    eval_json_path = os.path.join(ARTIFACTS_DIR, "evaluation_report.json")
    with open(eval_json_path, 'w', encoding='utf-8') as fp:
        json.dump(eval_results, fp, indent=2)
    print(f"\nSaved evaluation report: {eval_json_path}")

    # 4. Train Champion Daily Forecaster
    # Evaluate candidates: Random Forest vs Decision Tree vs Linear Regression vs Ridge
    comp = eval_results["model_comparison"]
    candidates = ["random_forest", "decision_tree", "linear_regression", "ridge"]
    best_type = min(candidates, key=lambda c: comp.get(c, {}).get("mean_mae", 999.0))
    best_mae = comp.get(best_type, {}).get("mean_mae", 0.0)

    print(f"\nChampion model selected: {best_type} (lowest walk-forward MAE: {best_mae:.2f} req/day)")

    feature_keys = [
        "day_of_week", "month", "day_of_month", "is_operating_day",
        "is_weekend", "is_peak_academic_season", "is_slow_academic_season",
        "is_month_start", "is_month_end", "lag_1", "lag_2", "lag_3",
        "lag_5", "lag_10", "rolling_mean_5", "rolling_mean_10",
        "rolling_std_5", "rolling_min_5", "rolling_max_5"
    ]

    X_full = np.array([[r[k] for k in feature_keys] for r in daily_features])
    y_full = np.array([r["target_requests"] for r in daily_features])

    forecaster = DailyDemandForecaster(model_type=best_type)
    forecaster.fit(X_full, y_full, feature_keys)

    forecaster_path = os.path.join(ARTIFACTS_DIR, "daily_forecaster.joblib")
    joblib.dump(forecaster, forecaster_path)
    print(f"Persisted forecaster artifact: {forecaster_path}")

    # 5. Fit Processing Time Turnaround Baseline
    proc_baseline = ProcessingTimeBaseline().fit(cleaned_requests)
    proc_path = os.path.join(ARTIFACTS_DIR, "processing_baseline.joblib")
    joblib.dump(proc_baseline, proc_path)
    print(f"Persisted processing baseline artifact: {proc_path}")

    # 6. Save Model Registry Metadata
    # Compute recent demand history for autoregressive warm start
    recent_history = [float(r["target_requests"]) for r in daily_features[-15:]]

    metadata = {
        "build_timestamp": datetime.now().isoformat(),
        "training_data_period": {
            "start": daily_records[0]["date"],
            "end": daily_records[-1]["date"],
            "total_operating_days": len(daily_records),
            "total_individual_requests": len(cleaned_requests)
        },
        "daily_forecaster": {
            "model_type": best_type,
            "version": forecaster.model_version,
            "feature_names": feature_keys,
            "residual_std": round(forecaster.residual_std, 2),
            "evaluation_metrics": comp.get(best_type, {})
        },
        "processing_time_baseline": {
            "version": "processing-time-baseline-v1",
            "basis": "administrative_standard_and_scheduled_lead_time",
            "is_empirical": False,
            "disclaimer": proc_baseline.disclaimer,
            "credential_days": proc_baseline.credential_turnarounds
        },
        "monthly_workload_thresholds": {
            "light_max": 500,
            "moderate_max": 850,
            "heavy_max": 1150,
            "very_heavy_min": 1150,
            "derivation_basis": "historical_2025_monthly_distribution"
        },
        "recent_history_buffer": recent_history
    }

    metadata_path = os.path.join(ARTIFACTS_DIR, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as fp:
        json.dump(metadata, fp, indent=2)
    print(f"Persisted metadata: {metadata_path}")
    print("=== Training & Persistence Complete ===")
    return metadata


if __name__ == "__main__":
    train_and_persist()

