"""
Processing Time Prediction and Turnaround Estimation for RegQuest ML.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional

ADMINISTRATIVE_STANDARD_DAYS: Dict[str, float] = {
    "TOR": 5.0,
    "HD": 3.0,
    "CAV": 3.0,
    "COE": 2.0,
    "GPA_CERT": 2.0,
    "GRAD_CERT": 2.0,
    "UNITS_CERT": 2.0,
    "ENROLLED_SUBJ_CERT": 2.0,
    "CAR": 3.0,
    "GRADES_CERT": 3.0,
    "CONVERSION_CERT": 2.0,
    "ENGLISH_CERT": 2.0,
    "GRADING_SYS_CERT": 2.0,
    "HONOR_CERT": 3.0,
    "GRADUATING_CERT": 2.0,
    "EVAL": 3.0,
    "F137": 5.0,
    "DIPLOMA": 7.0,
    "WES": 5.0,
    "STUDY_PERMIT": 2.0,
    "VERIFICATION": 3.0,
    "OTHER_CERT": 3.0,
    "DEFAULT": 3.0
}


def add_business_days(start_date: datetime, num_days: int) -> datetime:
    """Adds operating business days (skipping Sat & Sun) to a start date."""
    cur = start_date
    added = 0
    while added < num_days:
        cur += timedelta(days=1)
        if cur.weekday() < 5:  # Monday to Friday
            added += 1
    return cur


class ProcessingTimePredictor:
    """
    Predicts credential processing duration in operating days
    and computes the expected completion date.
    """
    def __init__(self, turnaround_standards: Dict[str, float] = None):
        self.standards = dict(turnaround_standards or ADMINISTRATIVE_STANDARD_DAYS)
        self.is_empirical = False
        self.disclaimer = (
            "Notice: Estimated processing duration reflects administrative registrar guidelines "
            "and scheduled turnaround. Actual empirical processing timestamps are not yet recorded "
            "in sufficient historical volume."
        )

    def predict_one(
        self,
        credential_type: str,
        purpose: str = "",
        num_docs: int = 1
    ) -> Dict[str, Any]:
        cred = (credential_type or "DEFAULT").strip().upper()
        base_days = self.standards.get(cred, self.standards.get("DEFAULT", 3.0))

        adjustment = 0.5 if num_docs > 1 else 0.0
        predicted_days = base_days + adjustment

        return {
            "predicted_processing_time": round(predicted_days, 1),
            "estimated_days": round(predicted_days, 1),
            "unit": "operating_days",
            "model_version": "processing-time-baseline-v1",
            "basis": "administrative_standard",
            "is_empirical": self.is_empirical,
            "disclaimer": self.disclaimer
        }

    def predict(
        self,
        credential_type: str,
        purpose: str = "Employment",
        date_requested: Optional[str] = None,
        num_documents: int = 1
    ) -> Dict[str, Any]:
        cred = (credential_type or "DEFAULT").strip().upper()
        base_days = self.standards.get(cred, self.standards.get("DEFAULT", 3.0))

        adjustment = 0.5 if num_documents > 1 else 0.0
        predicted_days = base_days + adjustment

        # Calculate estimated completion date
        start_dt = datetime.now()
        if date_requested:
            try:
                start_dt = datetime.strptime(date_requested[:10], "%Y-%m-%d")
            except Exception:
                pass

        completion_dt = add_business_days(start_dt, int(round(predicted_days)))

        return {
            "credential_type": cred,
            "purpose": purpose,
            "estimated_days": round(predicted_days, 1),
            "estimated_completion_date": completion_dt.strftime("%Y-%m-%d"),
            "confidence": 0.85,
            "confidence_interval_lower": max(1, int(round(predicted_days - 1.5))),
            "confidence_interval_upper": int(round(predicted_days + 2.0)),
            "unit": "operating_days",
            "is_empirical": self.is_empirical,
            "disclaimer": self.disclaimer,
            "source": "ml_service"
        }
