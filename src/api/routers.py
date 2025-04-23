
"""
API router aggregation
"""
from fastapi import APIRouter

from src.api.endpoints import prompts, optimization

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(prompts.router)
api_router.include_router(optimization.router)

# Add more routers as needed
# api_router.include_router(datasets.router)
# api_router.include_router(evaluation.router)
