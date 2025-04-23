"""
API router configuration
"""
from fastapi import APIRouter
from src.api.endpoints import prompts, optimization, cost_tracking

api_router = APIRouter()

api_router.include_router(prompts.router, prefix="/prompts", tags=["prompts"])
api_router.include_router(optimization.router, prefix="/optimization", tags=["optimization"])
api_router.include_router(cost_tracking.router, prefix="/cost_tracking", tags=["cost_tracking"])