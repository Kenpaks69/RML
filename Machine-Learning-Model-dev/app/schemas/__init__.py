"""Pydantic schemas for the RegQuest ML microservice."""

from .processing_time import ProcessingTimeRequest, ProcessingTimeResponse
from .forecasting import (
    HealthCheckResponse,
    DailyForecastRequest,
    DailyForecastItem,
    DailyForecastResponse,
    MonthlyWorkloadRequest,
    MonthlyWorkloadResponse,
)

__all__ = [
    "HealthCheckResponse",
    "ProcessingTimeRequest",
    "ProcessingTimeResponse",
    "DailyForecastRequest",
    "DailyForecastItem",
    "DailyForecastResponse",
    "MonthlyWorkloadRequest",
    "MonthlyWorkloadResponse",
]
