
"""
Prefect configuration settings for the workflow orchestration
"""
from prefect.settings import PREFECT_API_URL, PREFECT_API_KEY
from src.app.config import settings
import os

# Configure Prefect
if hasattr(settings, 'PREFECT_API_URL') and settings.PREFECT_API_URL:
    os.environ["PREFECT_API_URL"] = settings.PREFECT_API_URL
if hasattr(settings, 'PREFECT_API_KEY') and settings.PREFECT_API_KEY:
    os.environ["PREFECT_API_KEY"] = settings.PREFECT_API_KEY

# Default settings for local development
DEFAULT_QUEUE = "prompt-optimization"
DEFAULT_POOL = "prompt-pool"
DEFAULT_WORK_QUEUE_CONCURRENCY = 3
