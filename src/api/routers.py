
"""
API router aggregation
"""
from fastapi import APIRouter
from src.api.endpoints import prompts, optimization

# Create main API router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(prompts.router)
api_router.include_router(optimization.router)
