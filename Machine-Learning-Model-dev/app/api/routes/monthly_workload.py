"""
Monthly Workload Volume Forecasting Route.
"""

from fastapi import APIRouter, Depends, HTTPException
from ...schemas.forecasting import MonthlyWorkloadRequest, MonthlyWorkloadResponse
from ...dependencies import ModelManager, get_model_manager
from ...services.forecasting_service import ForecastingService

router = APIRouter(tags=["Monthly Workload Forecasting"])


@router.post("/forecast/monthly-workload", response_model=MonthlyWorkloadResponse)
def forecast_monthly_workload(
    req: MonthlyWorkloadRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    service = ForecastingService(manager)
    try:
        result = service.forecast_monthly_workload(target_month=req.target_month)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return MonthlyWorkloadResponse(
        status="success",
        month=result["month"],
        predicted_total_requests=result["predicted_total_requests"],
        operating_days=result["operating_days"],
        average_daily_requests=result["average_daily_requests"],
        workload_level=result["workload_level"],
        classification_basis=result["classification_basis"],
        model_version=result["model_version"],
        explanation=result["explanation"]
    )
