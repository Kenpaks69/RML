"""
RegQuest Automated Model Retraining, Validation, and Model Promotion Workflow.
Supports:
1. Ingesting and validating newly collected historical registrar records.
2. Chronological walk-forward re-evaluation of candidate models against the active champion model.
3. Automated rollback backup of previous production artifacts before model promotion.
"""

import os
import shutil
import json
from datetime import datetime
import joblib

from .pipeline import run_pipeline
from .features import extract_daily_features
from .models.forecaster import DailyDemandForecaster
from .models.processing_time import ProcessingTimeBaseline
from .evaluate import chronological_walk_forward_evaluation
from .train import load_daily_ts, load_cleaned_requests, ARTIFACTS_DIR

BACKUPS_DIR = os.path.join(ARTIFACTS_DIR, "backups")


def retrain_and_evaluate(force_promote: bool = False) -> dict:
    """
    Executes automated retraining pipeline with champion-challenger promotion gating.
    """
    os.makedirs(BACKUPS_DIR, exist_ok=True)
    print("=== RegQuest Model Retraining Workflow ===")

    # 1. Re-run data pipeline to integrate any newly added CSV files
    print("Executing data pipeline on all historical records...")
    run_pipeline()

    # 2. Load updated dataset & extract features
    daily_records = load_daily_ts()
    cleaned_requests = load_cleaned_requests()
    daily_features = extract_daily_features(daily_records)

    # 3. Chronological walk-forward evaluation on candidate models
    print(f"Evaluating candidate models across {len(daily_features)} daily records...")
    eval_results = chronological_walk_forward_evaluation(daily_features, n_splits=4, test_size=25)
    candidate_metrics = eval_results["model_comparison"]

    # 4. Read active production metadata
    meta_path = os.path.join(ARTIFACTS_DIR, "metadata.json")
    prod_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as fp:
            prod_meta = json.load(fp)

    active_model_type = prod_meta.get("daily_forecaster", {}).get("model_type", "ridge")
    active_mae = prod_meta.get("daily_forecaster", {}).get("evaluation_metrics", {}).get("mean_mae", 999.0)

    # 5. Determine best candidate model
    best_candidate_type = "ridge"
    lowest_mae = 999.0
    for m_type in ["poisson", "ridge", "gradient_boosting"]:
        mae = candidate_metrics.get(m_type, {}).get("mean_mae", 999.0)
        if mae < lowest_mae:
            lowest_mae = mae
            best_candidate_type = m_type

    print(f"\nActive Production Model: {active_model_type} (MAE: {active_mae:.2f})")
    print(f"Best Candidate Model:    {best_candidate_type} (MAE: {lowest_mae:.2f})")

    # 6. Promotion Gating
    # Promote if candidate achieves better or equivalent validated MAE (within 5% tolerance)
    promotion_eligible = (lowest_mae <= active_mae * 1.05) or force_promote

    promotion_result = {
        "timestamp": datetime.now().isoformat(),
        "active_model_type": active_model_type,
        "active_mae": active_mae,
        "candidate_model_type": best_candidate_type,
        "candidate_mae": lowest_mae,
        "promoted": False,
        "decision_reason": ""
    }

    if promotion_eligible:
        # Create timestamped rollback backup
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        curr_forecaster = os.path.join(ARTIFACTS_DIR, "daily_forecaster.joblib")
        if os.path.exists(curr_forecaster):
            backup_path = os.path.join(BACKUPS_DIR, f"daily_forecaster_{ts_tag}.joblib")
            shutil.copy2(curr_forecaster, backup_path)
            print(f"Preserved rollback backup: {backup_path}")

        # Train and save new champion model
        feature_keys = [
            "day_of_week", "month", "day_of_month", "is_operating_day",
            "is_weekend", "is_peak_academic_season", "is_slow_academic_season",
            "is_month_start", "is_month_end", "lag_1", "lag_2", "lag_3",
            "lag_5", "lag_10", "rolling_mean_5", "rolling_mean_10",
            "rolling_std_5", "rolling_min_5", "rolling_max_5"
        ]

        import numpy as np
        X_full = np.array([[r[k] for k in feature_keys] for r in daily_features])
        y_full = np.array([r["target_requests"] for r in daily_features])

        forecaster = DailyDemandForecaster(model_type=best_candidate_type)
        forecaster.fit(X_full, y_full, feature_keys)

        joblib.dump(forecaster, curr_forecaster)
        print(f"Promoted new model artifact to: {curr_forecaster}")

        # Update metadata
        recent_history = [float(r["target_requests"]) for r in daily_features[-15:]]
        updated_meta = {
            "build_timestamp": datetime.now().isoformat(),
            "training_data_period": {
                "start": daily_records[0]["date"],
                "end": daily_records[-1]["date"],
                "total_operating_days": len(daily_records),
                "total_individual_requests": len(cleaned_requests)
            },
            "daily_forecaster": {
                "model_type": best_candidate_type,
                "version": f"daily-forecaster-{best_candidate_type}-{ts_tag}",
                "feature_names": feature_keys,
                "residual_std": round(forecaster.residual_std, 2),
                "evaluation_metrics": candidate_metrics.get(best_candidate_type, {})
            },
            "processing_time_baseline": prod_meta.get("processing_time_baseline", {
                "version": "processing-time-baseline-v1",
                "basis": "administrative_standard"
            }),
            "monthly_workload_thresholds": prod_meta.get("monthly_workload_thresholds", {}),
            "recent_history_buffer": recent_history
        }

        with open(meta_path, 'w', encoding='utf-8') as fp:
            json.dump(updated_meta, fp, indent=2)

        promotion_result["promoted"] = True
        promotion_result["decision_reason"] = (
            f"Candidate model ({best_candidate_type}, MAE {lowest_mae:.2f}) passed validation criteria "
            f"and was promoted to production. Rollback artifact preserved."
        )
        print("Model promotion SUCCESSFUL.")
    else:
        promotion_result["promoted"] = False
        promotion_result["decision_reason"] = (
            f"Candidate model ({best_candidate_type}, MAE {lowest_mae:.2f}) failed to demonstrate "
            f"improvement over current production model (MAE {active_mae:.2f}). Promotion aborted."
        )
        print("Model promotion ABORTED: Candidate did not meet performance criteria.")

    return promotion_result


if __name__ == "__main__":
    retrain_and_evaluate()

