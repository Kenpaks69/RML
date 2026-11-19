"""
Centralized directory and file paths for the RegQuest ML package.
"""

from pathlib import Path

# Package root is D:\Capstone\ML\src\regquest_ml -> ML_ROOT is D:\Capstone\ML
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ML_ROOT = PACKAGE_ROOT.parent.parent

CONFIGS_DIR = ML_ROOT / "configs"
DATA_DIR = ML_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

ARTIFACTS_DIR = ML_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
DAILY_FORECAST_MODELS_DIR = MODELS_DIR / "daily_forecast"
PROCESSING_TIME_MODELS_DIR = MODELS_DIR / "processing_time"
METADATA_DIR = ARTIFACTS_DIR / "metadata"

REPORTS_DIR = ML_ROOT / "reports"
DATA_QUALITY_REPORTS_DIR = REPORTS_DIR / "data_quality"
EVALUATION_REPORTS_DIR = REPORTS_DIR / "evaluation"
FIGURES_REPORTS_DIR = REPORTS_DIR / "figures"

NOTEBOOKS_DIR = ML_ROOT / "notebooks"
