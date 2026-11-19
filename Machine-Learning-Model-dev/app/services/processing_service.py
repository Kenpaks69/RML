"""
Processing Time Domain Service.
Encapsulates turnaround duration estimation and business SLA logic.
"""

from typing import Dict, Any
from ..dependencies import ModelManager


class ProcessingService:
    def __init__(self, model_manager: ModelManager):
        self.manager = model_manager

    def predict_processing_time(
        self,
        credential_type: str,
        purpose: str = "Employment",
        num_documents: int = 1
    ) -> Dict[str, Any]:
        """
        Calculates predicted processing operating days and details using
        the fitted administrative baseline predictor.
        """
        if not self.manager.processing_baseline:
            raise RuntimeError("Processing time estimation model artifact is not loaded.")

        prediction = self.manager.processing_baseline.predict_one(
            credential_type=credential_type,
            purpose=purpose,
            num_docs=num_documents or 1
        )
        return prediction
