"""
Daily Demand Forecasting Route.
"""

from fastapi import APIRouter, Depends, HTTPException
from ...schemas.forecasting import DailyForecastRequest, DailyForecastResponse
from ...dependencies import ModelManager, get_model_manager
from ...services.forecasting_service import ForecastingService

router = APIRouter(tags=["Daily Demand Forecasting"])


@router.post("/forecast/daily", response_model=DailyForecastResponse)
def forecast_daily_demand(
    req: DailyForecastRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    service = ForecastingService(manager)
    try:
        result = service.forecast_daily_demand(
            start_date=req.start_date,
            number_of_operating_days=req.number_of_operating_days
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return DailyForecastResponse(
        status="success",
        forecast=result["forecast"],
        model_version=result["model_version"],
        start_date=result["start_date"],
        total_operating_days=result["total_operating_days"],
        total_predicted_requests=result["total_predicted_requests"]
    )
