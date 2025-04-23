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
"""
Configuration settings for the application
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # Environment and debugging
    ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "development")
    DEBUG: bool = os.environ.get("DEBUG", "1") == "1"
    
    # API configuration
    API_KEY: str = os.environ.get("API_KEY", "dev_key_change_me")
    ENFORCE_API_KEY: bool = os.environ.get("ENFORCE_API_KEY", "0") == "1"
    
    # Prefect configuration
    PREFECT_ENABLED: bool = os.environ.get("PREFECT_ENABLED", "0") == "1"
    PREFECT_API_URL: Optional[str] = os.environ.get("PREFECT_API_URL")
    PREFECT_API_KEY: Optional[str] = os.environ.get("PREFECT_API_KEY")
    
    # Vertex AI configuration
    VERTEX_PROJECT_ID: str = os.environ.get("VERTEX_PROJECT_ID", "")
    VERTEX_LOCATION: str = os.environ.get("VERTEX_LOCATION", "us-central1")
    
    # LLM configuration
    PRIMARY_MODEL: str = os.environ.get("PRIMARY_MODEL", "gemini-2.5-flash")
    OPTIMIZER_MODEL: str = os.environ.get("OPTIMIZER_MODEL", "gemini-2.5-pro")
    LLM_CACHE_ENABLED: bool = os.environ.get("LLM_CACHE_ENABLED", "1") == "1"
    
    # Application paths
    DATA_DIR: str = os.environ.get("DATA_DIR", "data")
    PROMPT_DIR: str = os.environ.get("PROMPT_DIR", "prompts")
    EXPERIMENT_DIR: str = os.environ.get("EXPERIMENT_DIR", "experiments")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()
