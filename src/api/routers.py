from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import datetime

from src.api.endpoints import (
    prompts,
    optimization,
    cost_tracking,
    experiments,
    datasets,
    inference
)

api_router = APIRouter(prefix="/api/v1")

# Include all endpoint routers
api_router.include_router(prompts.router)
api_router.include_router(optimization.router)
api_router.include_router(experiments.router)
api_router.include_router(datasets.router)
api_router.include_router(inference.router)
api_router.include_router(cost_tracking.router)


@api_router.get("/health")
async def health_check():
    """
    Health check endpoint for API
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    )


@api_router.get("/")
async def api_root():
    """
    API root endpoint with documentation
    """
    return JSONResponse(
        status_code=200,
        content={
            "name": "Prompt Optimization Platform API",
            "version": "1.0.0",
            "description": "API for optimizing LLM prompts using machine learning",
            "documentation": "/docs",
            "endpoints": [
                {"path": "/prompts", "description": "Prompt management"},
                {"path": "/optimization", "description": "Prompt optimization workflows"},
                {"path": "/experiments", "description": "Experiment management"},
                {"path": "/datasets", "description": "Dataset management"},
                {"path": "/inference", "description": "Model inference"},
                {"path": "/cost", "description": "Cost tracking"}
            ],
            "health": "/api/v1/health"
        }
    )
"""
Main API router that combines all endpoint modules
"""
from fastapi import APIRouter, Depends, HTTPException
from src.app.auth import get_api_key

from src.api.endpoints import (
    prompts,
    optimization,
    experiments,
    datasets,
    inference,
    cost_tracking,
    ml_settings
)

# Create main API router
api_router = APIRouter()

# Include each module's router with appropriate prefix
api_router.include_router(
    prompts.router,
    prefix="/prompts",
    tags=["Prompts"],
    dependencies=[Depends(get_api_key)]
)

api_router.include_router(
    optimization.router, 
    prefix="/optimize", 
    tags=["Optimization"],
    dependencies=[Depends(get_api_key)]
)

api_router.include_router(
    experiments.router,
    prefix="/experiments",
    tags=["Experiments"],
    dependencies=[Depends(get_api_key)]
)

api_router.include_router(
    datasets.router,
    prefix="/datasets",
    tags=["Datasets"],
    dependencies=[Depends(get_api_key)]
)

api_router.include_router(
    inference.router,
    prefix="/inference",
    tags=["Inference"],
    dependencies=[Depends(get_api_key)]
)

api_router.include_router(
    cost_tracking.router,
    prefix="/costs",
    tags=["Costs"],
    dependencies=[Depends(get_api_key)]
)

api_router.include_router(
    ml_settings.router,
    prefix="/ml-settings",
    tags=["ML Settings"],
    dependencies=[Depends(get_api_key)]
)
