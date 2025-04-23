
"""
API router configuration
"""
from fastapi import APIRouter

from src.api.endpoints import prompts, optimization

api_router = APIRouter()

api_router.include_router(prompts.router, prefix="/prompts", tags=["prompts"])
api_router.include_router(optimization.router, prefix="/optimization", tags=["optimization"])
