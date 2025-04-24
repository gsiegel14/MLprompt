
#!/usr/bin/env python3
"""
Set up Prefect with PostgreSQL for workflow orchestration
"""
import os
import sys
import subprocess
import logging
from pathlib import Path
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_prefect_installed():
    """Check if Prefect is installed"""
    try:
        subprocess.run(["prefect", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def create_prefect_profile():
    """Create a new Prefect profile for PostgreSQL"""
    logger.info("Creating Prefect profile for PostgreSQL")
    try:
        # Create a new profile
        subprocess.run(["prefect", "profile", "create", "postgresql-profile"], check=True)
        
        # Get database URL from environment
        db_url = os.getenv("PREFECT_API_DATABASE_CONNECTION_URL", 
                          "postgresql+asyncpg://prefect:prefectpass@localhost:5432/prefect")
        
        # Configure the profile
        subprocess.run([
            "prefect", "config", "set", 
            f"PREFECT_API_DATABASE_CONNECTION_URL={db_url}"
        ], check=True)
        
        # Set as active profile
        subprocess.run(["prefect", "profile", "use", "postgresql-profile"], check=True)
        
        logger.info("Prefect profile created and configured successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating Prefect profile: {str(e)}")
        return False

def create_work_queue():
    """Create a work queue for prompt optimization tasks"""
    from src.app.config.prefect_config import DEFAULT_QUEUE
    
    logger.info(f"Creating Prefect work queue: {DEFAULT_QUEUE}")
    try:
        subprocess.run(["prefect", "work-queue", "create", DEFAULT_QUEUE], check=True)
        logger.info(f"Work queue '{DEFAULT_QUEUE}' created successfully")
        return True
    except subprocess.CalledProcessError as e:
        if "already exists" in str(e.stderr):
            logger.info(f"Work queue '{DEFAULT_QUEUE}' already exists")
            return True
        logger.error(f"Error creating work queue: {str(e)}")
        return False

def main():
    """Main entry point for Prefect setup"""
    logger.info("Setting up Prefect with PostgreSQL")
    
    # Check if Prefect is installed
    if not check_prefect_installed():
        logger.error("Prefect is not installed. Please install it with 'pip install prefect'")
        sys.exit(1)
    
    # Create Prefect profile
    if not create_prefect_profile():
        logger.error("Failed to create Prefect profile")
        sys.exit(1)
    
    # Create work queue
    if not create_work_queue():
        logger.error("Failed to create work queue")
        sys.exit(1)
    
    logger.info("Prefect setup completed successfully")

if __name__ == "__main__":
    main()
