
"""
Factory function for creating the FastAPI application with Flask dashboard
"""
import os
import logging
from fastapi import FastAPI, APIRouter
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware
from flask import Flask

from src.api.routers import api_router
from src.app.config import settings
from src.app.dashboard import create_flask_app

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Application factory for creating integrated FastAPI/Flask app"""
    
    # Initialize FastAPI
    fast_app = FastAPI(
        title="Prompt Optimization Platform",
        description="ML-driven prompt optimization with Vertex AI and Prefect",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json"
    )
    
    # Add CORS middleware
    fast_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For development - restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register API routes
    fast_app.include_router(api_router, prefix="/api/v1")
    
    # Create Flask dashboard
    flask_app = create_flask_app()
    
    # Mount Flask app under the /dashboard path
    fast_app.mount("/dashboard", WSGIMiddleware(flask_app))
    
    # Health check endpoint
    @fast_app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "version": "1.0.0"}
    
    # Configure startup and shutdown handlers
    @fast_app.on_event("startup")
    async def startup_event():
        logger.info("Starting Prompt Optimization Platform API")
        
        # Initialize Prefect client if needed
        if hasattr(settings, 'PREFECT_ENABLED') and settings.PREFECT_ENABLED:
            try:
                from prefect.client import get_client
                async with get_client() as client:
                    # Check Prefect health
                    healthcheck = await client.api_healthcheck()
                    logger.info(f"Prefect API health check: {healthcheck}")
            except Exception as e:
                logger.error(f"Failed to connect to Prefect: {e}")
        
    @fast_app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down Prompt Optimization Platform API")
    
    return fast_app
