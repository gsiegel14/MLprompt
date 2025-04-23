import os
from app import app
from src.app.utils.logger import configure_logging
import logging

# Configure advanced logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, log_level, logging.INFO)
configure_logging(level=numeric_level)

# Create logger for this module
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Starting Prompt Optimization Platform")
    logger.info(f"Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    logger.info(f"Log level: {log_level}")
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "True").lower() == "true"
    
    logger.info(f"Starting server on port {port}, debug mode: {debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)