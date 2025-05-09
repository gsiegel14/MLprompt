"""
FastAPI app entry point
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from src.api.routers import api_router
from src.app.config import settings
from src.app.factory import create_flask_app

# Configure logger
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Prompt Optimization Platform API",
    description="API for ML-driven prompt optimization workflow",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "version": app.version,
        "config": {
            "primary_model": settings.PRIMARY_MODEL_NAME,
            "optimizer_model": settings.OPTIMIZER_MODEL_NAME
        }
    }

# Include API routers
app.include_router(api_router, prefix="/api/v1")

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting Prompt Optimization Platform API v{app.version}")

    # Check if Vertex AI project is configured
    if not settings.VERTEX_PROJECT_ID:
        logger.warning("VERTEX_PROJECT_ID not set - some features may not work properly")

    # Initialize any required services
    # This could include setting up database connections, etc.

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Prompt Optimization Platform API")

    # Clean up any resources
"""
Application entry point for the Prompt Optimization Platform
"""
import os
import logging
from pathlib import Path
from src.app.factory import create_app
from src.app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create the application
app = create_app()

# Make the app directly runnable
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "src.app.main:app",
        host="0.0.0.0",
        port=5000,
        reload=settings.DEBUG
    )