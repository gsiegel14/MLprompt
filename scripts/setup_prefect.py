
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
        # Check if we're running on Replit
        is_replit = "REPL_ID" in os.environ and "REPLIT_DB_URL" in os.environ
        
        # Get main database URL
        if is_replit and os.getenv("REPLIT_DB_URL"):
            main_db_url = os.getenv("DATABASE_URL")
            logger.info("Using Replit PostgreSQL database for Prefect")
            
            # For Replit, we use same database with a schema
            prefect_db_url = main_db_url
            
            # Make sure the schema exists
            try:
                conn = psycopg2.connect(main_db_url)
                cursor = conn.cursor()
                cursor.execute("CREATE SCHEMA IF NOT EXISTS prefect")
                conn.commit()
                cursor.close()
                conn.close()
                logger.info("Ensured 'prefect' schema exists in Replit PostgreSQL")
            except Exception as e:
                logger.warning(f"Error creating prefect schema: {str(e)}")
        else:
            main_db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/promptopt")
            # Create a prefect-specific database URL (same server, different database)
            parts = main_db_url.rsplit('/', 1)
            base_url = parts[0]
            prefect_db_url = f"{base_url}/prefect"
        
        # Set environment variable for Prefect
        os.environ["PREFECT_API_DATABASE_CONNECTION_URL"] = prefect_db_url
        
        # Create profile name based on environment
        profile_name = "replit-postgres-profile" if is_replit else "postgres-profile"
        
        # Create a new profile
        subprocess.run(["prefect", "profile", "create", profile_name], check=True)
        
        # Configure the profile
        subprocess.run([
            "prefect", "config", "set", 
            f"PREFECT_API_DATABASE_CONNECTION_URL={prefect_db_url}"
        ], check=True)
        
        # Set as active profile
        subprocess.run(["prefect", "profile", "use", profile_name], check=True)
        
        # Set API URL to bind to all interfaces so it's accessible from Replit
        subprocess.run([
            "prefect", "config", "set",
            "PREFECT_API_URL=http://0.0.0.0:4200/api"
        ], check=True)
        
        logger.info(f"Prefect profile created and configured to use PostgreSQL at {prefect_db_url}")
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
