"""
Script: evaluate_models.py
Executes chronological walk-forward cross-validation comparing:
1. Random Forest Regressor (Champion)
2. Decision Tree Regressor
3. Linear Regression
4. Rolling 5-day Mean Baseline
5. Previous Day Baseline
6. Historical Mean Baseline

Outputs comprehensive comparison report to reports/evaluation/ and artifacts/.
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
    EVALUATION_REPORTS_DIR,
    ARTIFACTS_DIR,
)
from src.regquest_ml.data.loader import load_raw_csv_files, load_processed_data
from src.regquest_ml.data.cleaner import clean_raw_records
from src.regquest_ml.features.daily_demand import (
    aggregate_daily_time_series,
    build_daily_features_table,
)
from src.regquest_ml.evaluation.forecasting_metrics import (
    chronological_walk_forward_evaluation,
    summarize_metrics,
)


def main():
    print("=" * 70)
    print("Starting Chronological Walk-Forward Cross-Validation (4 Splits)")
    print("=" * 70)

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

    daily_features = build_daily_features_table(daily_series)
    print(f"Feature table observations: {len(daily_features)}")


    # 2. Walk-forward CV
    cv_output = chronological_walk_forward_evaluation(daily_features, n_splits=4, test_size=25)
    summary = summarize_metrics(cv_output["models"])

    print("\n" + "=" * 70)
    print(f"{'Model':<26} | {'MAE':<8} | {'MedAE':<8} | {'RMSE':<8} | Status")
    print("-" * 70)

    models_order = [
        ("random_forest", "Random Forest Regressor (Champion)"),
        ("decision_tree", "Decision Tree Regressor"),
        ("linear_regression", "Linear Regression"),
        ("rolling_5d_mean", "Rolling 5-Day Average Baseline"),
        ("previous_day", "Previous Operating Day Baseline"),
        ("historical_mean", "Historical Mean Baseline"),
    ]

    for model_key, display_name in models_order:
        stats = summary.get(model_key, {})
        mae = stats.get("mae_mean", 0.0)
        medae = stats.get("medae_mean", 0.0)
        rmse = stats.get("rmse_mean", 0.0)
        status = "CHAMPION" if model_key == "random_forest" else "Benchmark"
        print(f"{display_name:<26} | {mae:<8.2f} | {medae:<8.2f} | {rmse:<8.2f} | {status}")

    print("=" * 70)

    # 3. Persist evaluation report
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_splits": 4,
        "test_horizon_days": 25,
        "champion_model": "random_forest",
        "scoreboard": summary,
        "split_details": cv_output.get("split_details", [])
    }

    report_path = EVALUATION_REPORTS_DIR / "evaluation_report.json"
    legacy_path = ARTIFACTS_DIR / "evaluation_report.json"
    os.makedirs(EVALUATION_REPORTS_DIR, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    with open(legacy_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print(f"\nEvaluation reports saved to:")
    print(f"  -> {report_path}")
    print(f"  -> {legacy_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
