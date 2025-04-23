
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
    
    # Add redirect from root to dashboard
    @fast_app.get("/", include_in_schema=False)
    async def root_redirect():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/dashboard")
    
    # Health check endpoint
    @fast_app.get("/health", tags=["Health"])
    async def health_check():
        """API health check endpoint"""
        components_status = {
            "api": "active",
            "database": "active" if hasattr(settings, "DATABASE_URL") else "inactive",
            "prefect": "active" if settings.PREFECT_ENABLED else "inactive",
            "llm_service": "active",
            "cache": "active" if settings.LLM_CACHE_ENABLED else "inactive"
        }
        
        return {
            "status": "ok", 
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT,
            "components": components_status
        }
    
    # API documentation endpoints
    @fast_app.get("/docs", include_in_schema=False)
    async def docs_redirect():
        """Redirect /docs to /api/docs"""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/api/docs")
    
    @fast_app.get("/redoc", include_in_schema=False)
    async def redoc_redirect():
        """Redirect /redoc to /api/redoc"""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/api/redoc")
    
    # Configure startup and shutdown handlers
    @fast_app.on_event("startup")
    async def startup_event():
        """Application startup handler"""
        logger.info("Starting Prompt Optimization Platform API")
        
        # Create necessary directories
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        os.makedirs(settings.PROMPT_DIR, exist_ok=True)
        os.makedirs(settings.EXPERIMENT_DIR, exist_ok=True)
        
        # Initialize Prefect client if enabled
        if settings.PREFECT_ENABLED:
            try:
                from prefect.client import get_client
                async with get_client() as client:
                    # Check Prefect health
                    healthcheck = await client.api_healthcheck()
                    logger.info(f"Prefect API health check: {healthcheck}")
            except Exception as e:
                logger.error(f"Failed to connect to Prefect: {e}")
                logger.error(f"Prefect error details: {str(e)}")
        
        # Initialize unified logging
        from src.app.utils.logger import setup_unified_logging
        setup_unified_logging()
        
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"Debug mode: {'enabled' if settings.DEBUG else 'disabled'}")
        logger.info(f"Vertex AI project: {settings.VERTEX_PROJECT_ID}")
        logger.info(f"API startup complete")
        
    @fast_app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown handler"""
        logger.info("Shutting down Prompt Optimization Platform API")
        
        # Perform cleanup operations
        try:
            # Close any active connections
            pass
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
    
    return fast_app
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware
import logging
from src.api.routers import api_router
from src.app.dashboard import create_flask_app
from src.app.config import settings

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Application factory for creating the FastAPI app with Flask dashboard integration"""
    
    # Initialize FastAPI
    app = FastAPI(
        title="Prompt Optimization Platform",
        description="ML-driven prompt optimization with Vertex AI and Prefect",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register API routes
    app.include_router(api_router, prefix="/api/v1")
    
    # Create Flask dashboard app
    flask_app = create_flask_app()
    
    # Mount Flask app at /dashboard
    app.mount("/dashboard", WSGIMiddleware(flask_app))
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok"}
    
    # Root redirect to dashboard
    @app.get("/", tags=["Root"])
    async def root():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/dashboard")
    
    return app
