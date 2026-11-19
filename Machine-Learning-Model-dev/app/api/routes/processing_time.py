"""
Processing Time Prediction Route.
"""

from fastapi import APIRouter, Depends, HTTPException
from ...schemas.processing_time import ProcessingTimeRequest, ProcessingTimeResponse
from ...dependencies import ModelManager, get_model_manager
from ...services.processing_service import ProcessingService

router = APIRouter(tags=["Processing Time"])


@router.post("/predict/processing-time", response_model=ProcessingTimeResponse)
def predict_processing_time(
    req: ProcessingTimeRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    service = ProcessingService(manager)
    try:
        prediction = service.predict_processing_time(
            credential_type=req.credential_type,
            purpose=req.purpose or "Employment",
            num_documents=req.num_documents or 1
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ProcessingTimeResponse(
        status="success",
        predicted_processing_time=prediction["predicted_processing_time"],
        unit=prediction["unit"],
        model_version=prediction["model_version"],
        basis=prediction["basis"],
        is_empirical=prediction["is_empirical"],
        disclaimer=prediction.get("disclaimer")
    )
