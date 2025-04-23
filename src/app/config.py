"""
Configuration settings for the prompt optimization platform
"""
from pydantic_settings import BaseSettings
from typing import Optional, List
import os
import secrets

class Settings(BaseSettings):
    # Vertex AI Settings
    VERTEX_PROJECT_ID: str
    VERTEX_LOCATION: str = "us-central1"
    PRIMARY_MODEL_NAME: str = "gemini-1.5-flash-001"
    OPTIMIZER_MODEL_NAME: str = "gemini-1.5-pro-001"
    
    # Optional integrations
    WANDB_API_KEY: Optional[str] = None
    
    # Storage Settings
    GCS_BUCKET_NAME: Optional[str] = None
    
    # Prefect Settings
    PREFECT_API_URL: Optional[str] = None
    PREFECT_API_KEY: Optional[str] = None
    
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