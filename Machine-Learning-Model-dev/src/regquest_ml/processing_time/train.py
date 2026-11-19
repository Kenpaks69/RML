"""
Processing Time Baseline Model Fitting and Serialization.
"""

import os
from collections import defaultdict
from typing import List, Dict, Any, Optional
import numpy as np
import joblib

from ..common.paths import (
    PROCESSING_TIME_MODELS_DIR,
    ARTIFACTS_DIR
)
from .predict import ProcessingTimePredictor, ADMINISTRATIVE_STANDARD_DAYS


def train_processing_time_baseline(
    cleaned_records: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> ProcessingTimePredictor:
    """
    Fits median scheduled turnaround business days from cleaned records.
    Serializes fitted predictor to output_path.
    """
    standards = dict(ADMINISTRATIVE_STANDARD_DAYS)
    grouped = defaultdict(list)

    for r in cleaned_records:
        cred = r.get("primary_credential", "DEFAULT")
        days = r.get("scheduled_lead_business_days")
        if days is not None and 1 <= float(days) <= 30:
            grouped[cred].append(float(days))

    for cred, vals in grouped.items():
        if len(vals) >= 10:
            standards[cred] = float(np.median(vals))

    predictor = ProcessingTimePredictor(turnaround_standards=standards)

    # Serialize to both the new models directory and the legacy root artifacts dir
    save_target = output_path or str(PROCESSING_TIME_MODELS_DIR / "processing_baseline.joblib")
    os.makedirs(os.path.dirname(save_target), exist_ok=True)
    joblib.dump(predictor, save_target)

    legacy_target = str(ARTIFACTS_DIR / "processing_baseline.joblib")
    joblib.dump(predictor, legacy_target)

    return predictor
