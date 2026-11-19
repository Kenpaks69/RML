"""
Feature Extractors for Individual Document Processing Turnaround Estimation.
"""

from typing import Dict, Any


def extract_request_features(
    credential_code: str,
    purpose: str = "Employment",
    num_documents: int = 1,
    day_of_week: int = 0
) -> Dict[str, Any]:
    """
    Extracts features for document turnaround duration estimation.
    """
    return {
        "credential_code": str(credential_code).upper().strip(),
        "purpose": str(purpose).strip(),
        "num_documents": max(1, int(num_documents)),
        "is_multi_document": int(num_documents > 1),
        "day_of_week": int(day_of_week)
    }
