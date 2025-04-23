"""
Configuration settings for the prompt optimization platform
"""
from pydantic_settings import BaseSettings
from typing import Optional, List
import os
import secrets

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Prompt Optimization API"

    # Google Vertex AI Settings
    VERTEX_PROJECT_ID: str = os.getenv("VERTEX_PROJECT_ID", "")
    VERTEX_LOCATION: str = os.getenv("VERTEX_LOCATION", "us-central1")
    PRIMARY_MODEL_NAME: str = os.getenv("PRIMARY_MODEL_NAME", "gemini-1.5-flash")
    OPTIMIZER_MODEL_NAME: str = os.getenv("OPTIMIZER_MODEL_NAME", "gemini-1.5-pro")

    # Prefect Settings
    PREFECT_API_URL: str = os.getenv("PREFECT_API_URL", "")
    PREFECT_API_KEY: str = os.getenv("PREFECT_API_KEY", "")
    GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "")

    # Optional integrations
    WANDB_API_KEY: Optional[str] = None

    # Storage Settings


    # Application Settings
    DEBUG: bool = False
    MAX_EXAMPLES: int = 100
    DEFAULT_METRICS: List[str] = ["exact_match", "semantic_similarity"]
    OPTIMIZATION_STRATEGIES: List[str] = ["balanced", "accuracy", "creativity"]

    # Token Efficiency Settings
    ENABLE_CACHING: bool = True
    CACHE_MAX_AGE_SECONDS: int = 3600  # 1 hour
    CACHE_MAX_ENTRIES: int = 1000
    TRACK_COSTS: bool = True
    COST_REPORTS_DIR: str = "cost_reports"

    # Authentication Settings
    JWT_SECRET_KEY: str = secrets.token_urlsafe(32)  # Generate a random secret key for development
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Create cost reports directory if it doesn't exist
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not os.path.exists(self.COST_REPORTS_DIR):
            os.makedirs(self.COST_REPORTS_DIR)

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

settings = Settings()