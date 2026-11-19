"""
Backward-compatibility re-export module for app.schemas.
"""

from .schemas.processing_time import ProcessingTimeRequest, ProcessingTimeResponse
from .schemas.forecasting import (
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
