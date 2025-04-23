
"""
Configuration settings for the Prompt Optimization Platform
"""
from pydantic import BaseSettings, Field
from typing import Optional
import os
import yaml
import logging

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file
    """
    # Vertex AI settings
    VERTEX_PROJECT_ID: Optional[str] = None
    VERTEX_LOCATION: str = "us-central1"
    PRIMARY_MODEL_NAME: str = "gemini-1.5-flash-001"
    OPTIMIZER_MODEL_NAME: str = "gemini-1.5-pro-001"
    
    # Optional integrations
    WANDB_API_KEY: Optional[str] = None
    GCS_BUCKET_NAME: Optional[str] = None
    
    # Prefect settings
    PREFECT_API_URL: Optional[str] = None
    PREFECT_API_KEY: Optional[str] = None
    
    # App settings
    LOG_LEVEL: str = "INFO"
    MAX_ITERATIONS: int = 10
    TARGET_THRESHOLD: float = 0.90
    PATIENCE: int = 3
    DEFAULT_METRICS: list = Field(default_factory=lambda: ["exact_match", "semantic_similarity"])
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

# Load from YAML config if available
def load_config_from_yaml(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        return {}
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        return {}

# Create settings instance
yaml_config = load_config_from_yaml()
settings = Settings(**yaml_config)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
