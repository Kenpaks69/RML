"""
Model Artifact Manager and Dependency Injection Provider for FastAPI.
Loads serialized models and metadata at startup with graceful fallback.
"""

import os
import sys
import json
import time
from typing import Optional, Dict, Any
import joblib

from .config import ARTIFACTS_DIR

BASE_ML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_ML_DIR not in sys.path:
    sys.path.insert(0, BASE_ML_DIR)

START_TIME = time.time()


class ModelManager:
    _instance = None

    def __init__(self):
        self.daily_forecaster = None
        self.processing_baseline = None
        self.metadata = {}
        self.is_loaded = False
        self.load_artifacts()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def load_artifacts(self):
        meta_paths = [
            os.path.join(ARTIFACTS_DIR, "metadata", "metadata.json"),
            os.path.join(ARTIFACTS_DIR, "metadata.json")
        ]
        forecaster_paths = [
            os.path.join(ARTIFACTS_DIR, "models", "daily_forecast", "daily_forecaster.joblib"),
            os.path.join(ARTIFACTS_DIR, "daily_forecaster.joblib")
        ]
        baseline_paths = [
            os.path.join(ARTIFACTS_DIR, "models", "processing_time", "processing_baseline.joblib"),
            os.path.join(ARTIFACTS_DIR, "processing_baseline.joblib")
        ]

        for mp in meta_paths:
            if os.path.exists(mp):
                try:
                    with open(mp, 'r', encoding='utf-8') as fp:
                        self.metadata = json.load(fp)
                    break
                except Exception as e:
                    print(f"Warning: Failed to load metadata from {mp}: {e}")

        for fp in forecaster_paths:
            if os.path.exists(fp):
                try:
                    self.daily_forecaster = joblib.load(fp)
                    break
                except Exception as e:
                    print(f"Warning: Failed to load daily forecaster from {fp}: {e}")

        for bp in baseline_paths:
            if os.path.exists(bp):
                try:
                    self.processing_baseline = joblib.load(bp)
                    break
                except Exception as e:
                    print(f"Warning: Failed to load processing baseline from {bp}: {e}")

        self.is_loaded = bool(self.daily_forecaster and self.processing_baseline)


    def get_status(self) -> Dict[str, Any]:
        return {
            "daily_forecaster_available": self.daily_forecaster is not None,
            "processing_baseline_available": self.processing_baseline is not None,
            "metadata_available": bool(self.metadata),
            "versions": {
                "daily_forecaster": self.metadata.get("daily_forecaster", {}).get("version", "unknown"),
                "processing_time": self.metadata.get("processing_time_baseline", {}).get("version", "unknown")
            },
            "uptime_seconds": round(time.time() - START_TIME, 1)
        }


def get_model_manager() -> ModelManager:
    return ModelManager.get_instance()

