"""
Health Check and Service Status Route.
"""

from fastapi import APIRouter, Depends
from ...schemas.forecasting import HealthCheckResponse
from ...dependencies import ModelManager, get_model_manager
from ...config import PROJECT_NAME, VERSION

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthCheckResponse)
def health_check(manager: ModelManager = Depends(get_model_manager)):
    status_info = manager.get_status()
    all_ready = (
        status_info["daily_forecaster_available"] and
        status_info["processing_baseline_available"]
    )

    return HealthCheckResponse(
        status="ok" if all_ready else "degraded",
        project_name=PROJECT_NAME,
        service_version=VERSION,
        model_availability={
            "daily_forecaster": status_info["daily_forecaster_available"],
            "processing_baseline": status_info["processing_baseline_available"]
        },
        model_versions=status_info["versions"],
        uptime_seconds=status_info["uptime_seconds"]
    )
