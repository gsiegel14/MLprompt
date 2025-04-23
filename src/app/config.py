"""
Configuration settings for the prompt optimization platform
"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    VERTEX_PROJECT_ID: str
    VERTEX_LOCATION: str = "us-central1"
    PRIMARY_MODEL_NAME: str = "gemini-1.5-flash-001"
    OPTIMIZER_MODEL_NAME: str = "gemini-1.5-pro-001"
    WANDB_API_KEY: Optional[str] = None
    GCS_BUCKET_NAME: Optional[str] = None
    PREFECT_API_URL: Optional[str] = None
    PREFECT_API_KEY: Optional[str] = None

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

settings = Settings()