"""
Processing Time Turnaround Predictor and Baseline Estimator.
Implements administrative standard baselines and scheduled duration regression
with explicit disclaimers regarding the historical lack of empirical physical completion timestamps.
"""

from typing import Dict, Any, Tuple
import numpy as np

# Administrative defaults from university registrar guidelines and backend CredentialType configuration
ADMINISTRATIVE_STANDARD_DAYS = {
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
    "DEFAULT": 3.0
}


class ProcessingTimeBaseline:
    """
    Defensible turnaround duration baseline based on registrar standards
    and observed scheduled turnaround times.
    """
    def __init__(self):
        self.credential_turnarounds = dict(ADMINISTRATIVE_STANDARD_DAYS)
        self.is_empirical = False
        self.disclaimer = (
            "Notice: Estimated processing duration reflects administrative registrar guidelines "
            "and scheduled turnaround. Actual empirical processing timestamps are not yet recorded "
            "in sufficient historical volume."
        )

    def fit(self, cleaned_records: list):
        """Calculates observed median scheduled business days by credential type."""
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in cleaned_records:
            cred = r.get("primary_credential", "DEFAULT")
            days = r.get("scheduled_lead_business_days")
            if days is not None and 1 <= days <= 30:
                grouped[cred].append(days)
                
        for cred, vals in grouped.items():
            if len(vals) >= 10:
                self.credential_turnarounds[cred] = float(np.median(vals))
        return self

    def predict_one(self, credential_type: str, purpose: str = "", num_docs: int = 1) -> Dict[str, Any]:
        cred = (credential_type or "DEFAULT").strip().upper()
        base_days = self.credential_turnarounds.get(cred, self.credential_turnarounds.get("DEFAULT", 3.0))
        
        # Add half a business day if multi-document package
        adjustment = 0.5 if num_docs > 1 else 0.0
        predicted_days = base_days + adjustment
        
        return {
            "predicted_processing_time": round(predicted_days, 1),
            "unit": "operating_days",
            "model_version": "processing-time-baseline-v1",
            "basis": "administrative_standard",
            "is_empirical": False,
            "disclaimer": self.disclaimer
        }

