"""
Configuration settings for the prompt optimization platform
"""
from pydantic_settings import BaseSettings
from typing import Optional

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

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

settings = Settings()