"""
Daily Demand and Monthly Workload Forecasting Pydantic Schemas.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator
import re


class HealthCheckResponse(BaseModel):
    status: str = "ok"
    project_name: str
    service_version: str
    model_availability: Dict[str, bool]
    model_versions: Dict[str, str]
    uptime_seconds: float


class DailyForecastRequest(BaseModel):
    start_date: str = Field(
        ...,
        description="Forecast starting date in YYYY-MM-DD format"
    )
    number_of_operating_days: int = Field(
        20,
        ge=1,
        le=60,
        description="Number of future operating days to forecast"
    )

    @field_validator("start_date")
    @classmethod
    def validate_date(cls, v):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("start_date must be in YYYY-MM-DD format")
        return v


class DailyForecastItem(BaseModel):
    date: str
    weekday: str
    predicted_requests: int
    lower_bound_90: int
    upper_bound_90: int


class DailyForecastResponse(BaseModel):
    status: str = "success"
    forecast: List[DailyForecastItem]
    model_version: str
    start_date: str
    total_operating_days: int
    total_predicted_requests: int


class MonthlyWorkloadRequest(BaseModel):
    target_month: Optional[str] = Field(
        None,
        description="Target month in YYYY-MM format (e.g. 2026-12)"
    )

    @field_validator("target_month")
    @classmethod
    def validate_month(cls, v):
        if v and not re.match(r"^\d{4}-\d{2}$", v):
            raise ValueError("target_month must be in YYYY-MM format")
        return v


class MonthlyWorkloadResponse(BaseModel):
    status: str = "success"
    month: str
    predicted_total_requests: int
    operating_days: int
    average_daily_requests: float
    workload_level: str  # Light, Moderate, Heavy, Very Heavy
    classification_basis: str
    model_version: str
    explanation: str
