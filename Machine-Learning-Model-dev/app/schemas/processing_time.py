"""
Processing Time Pydantic Validation and Response Schemas.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator
import re


class ProcessingTimeRequest(BaseModel):
    credential_type: str = Field(
        ...,
        description="Short code or name of credential (e.g. TOR, HD, CAV, COE, CAR)"
    )
    purpose: Optional[str] = Field(
        "Employment",
        description="Purpose of request (e.g. Employment, Evaluation, Scholarship)"
    )
    date_requested: Optional[str] = Field(
        None,
        description="Request submission date in YYYY-MM-DD format"
    )
    num_documents: Optional[int] = Field(
        1,
        ge=1,
        le=20,
        description="Number of document types or copies requested"
    )

    @field_validator("date_requested")
    @classmethod
    def validate_date(cls, v):
        if v and not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("date_requested must be in YYYY-MM-DD format")
        return v


class ProcessingTimeResponse(BaseModel):
    status: str = "success"
    predicted_processing_time: float
    unit: str = "operating_days"
    model_version: str
    basis: str
    is_empirical: bool
    disclaimer: Optional[str] = None
