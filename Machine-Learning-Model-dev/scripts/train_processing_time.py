"""
Script: train_processing_time.py
Trains administrative baseline for credential processing turnaround time.
Serializes artifact to artifacts/models/processing_time/ and legacy artifacts/.
"""

import sys
import os
from pathlib import Path

# Add ML root to path
ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from src.regquest_ml.common.paths import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    PROCESSING_TIME_MODELS_DIR,
    ARTIFACTS_DIR,
)
from src.regquest_ml.data.loader import load_raw_csv_files, load_processed_data
from src.regquest_ml.data.cleaner import clean_raw_records
from src.regquest_ml.processing_time.train import train_processing_time_baseline


def main():
    print("=" * 60)
    print("Starting Credential Processing Time Baseline Training")
    print("=" * 60)

    # 1. Load data
    processed_path = PROCESSED_DATA_DIR / "cleaned_requests.csv"
    if processed_path.exists():
        print(f"Loading cleaned records from: {processed_path}")
        records = load_processed_data(str(processed_path))
    else:
        print(f"Processing raw datasets from: {RAW_DATA_DIR}")
        raw_records, _ = load_raw_csv_files(str(RAW_DATA_DIR))
        cleaning_result = clean_raw_records(raw_records)
        records = cleaning_result["cleaned"]

    print(f"Total records available: {len(records):,}")

    # 2. Train baseline
    predictor = train_processing_time_baseline(records)

    target_path = PROCESSING_TIME_MODELS_DIR / "processing_baseline.joblib"
    legacy_path = ARTIFACTS_DIR / "processing_baseline.joblib"

    print(f"Model saved successfully to:")
    print(f"  -> {target_path}")
    print(f"  -> {legacy_path}")
    print("\nTurnaround standards sample:")
    for cred, days in list(predictor.standards.items())[:6]:
        print(f"  - {cred}: {days} operating days")
    print("=" * 60)


if __name__ == "__main__":
    main()
