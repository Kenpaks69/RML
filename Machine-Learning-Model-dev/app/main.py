"""
RegQuest Machine Learning Microservice.
FastAPI Application Entrypoint.
Provides endpoints for:
1. Document processing turnaround duration prediction (/predict/processing-time)
2. Daily incoming credential request forecasting (/forecast/daily)
3. Monthly workload volume forecasting and classification (/forecast/monthly-workload)
4. Health check and model readiness (/health)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import PROJECT_NAME, VERSION
from .dependencies import ModelManager
from .api.routes import (
    health_router,
    processing_time_router,
    daily_forecast_router,
    monthly_workload_router
)

app = FastAPI(
    title=PROJECT_NAME,
    description="Machine Learning Microservice for RegQuest Credential Management System",
    version=VERSION,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers from app.api.routes
app.include_router(health_router)
app.include_router(processing_time_router)
app.include_router(daily_forecast_router)
app.include_router(monthly_workload_router)

# Preload models
ModelManager.get_instance()


@app.get("/", tags=["Root"])
def root():
    return {
        "service": PROJECT_NAME,
        "version": VERSION,
        "docs_url": "/docs",
        "health_url": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
