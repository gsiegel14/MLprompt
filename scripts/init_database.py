
#!/usr/bin/env python
"""
Initialize the database for the ML Prompt Optimization Platform.
Creates required tables and populates initial data.
"""
import os
import sys
import logging
from pathlib import Path
import importlib.util
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from alembic.config import Config
from alembic import command

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import database models
from src.app.models.database_models import (
    Prompt, Dataset, Experiment, MetricsRecord, ModelConfiguration,
    Base
)
from src.app.database.db import engine, SessionLocal
from src.app.repositories.ml_settings_repository import MLSettingsRepository

def check_database_exists():
    """Check if the database exists and create it if it doesn't"""
    # Check if we're running on Replit
    is_replit = "REPL_ID" in os.environ and "REPLIT_DB_URL" in os.environ
    
    # Get PostgreSQL URL from environment or use default
    db_url = os.getenv("DATABASE_URL")
    
    if is_replit and os.getenv("REPLIT_DB_URL"):
        # Using Replit's PostgreSQL
        logger.info("Running on Replit with built-in PostgreSQL database")
        
        try:
            # Test connection to the database
            conn = psycopg2.connect(db_url)
            conn.close()
            logger.info("Successfully connected to Replit PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Replit PostgreSQL: {str(e)}")
            logger.error("Please create a PostgreSQL database in the Replit Database panel")
            logger.error("1. Open a new tab in Replit and type 'Database'")
            logger.error("2. Click 'Create a database' button")
            return False
    
    # Standard PostgreSQL setup for non-Replit environments
    if not db_url or not db_url.startswith("postgresql"):
        db_url = "postgresql://postgres:postgres@localhost:5432/promptopt"
        os.environ["DATABASE_URL"] = db_url
    
    try:
        # Extract database name from URL
        db_name = db_url.split("/")[-1]
        # Create a connection to PostgreSQL server
        conn_str = db_url.rsplit("/", 1)[0] + "/postgres"
        conn = psycopg2.connect(conn_str)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        if cursor.fetchone() is None:
            # Create database if it doesn't exist
            logger.info(f"Database {db_name} does not exist, creating it...")
            cursor.execute(f"CREATE DATABASE {db_name}")
            logger.info(f"Database {db_name} created successfully")
        else:
            logger.info(f"Database {db_name} already exists")
        
        cursor.close()
        conn.close()
        
        # Test connection to the actual database
        test_conn = psycopg2.connect(db_url)
        test_conn.close()
        logger.info("Successfully connected to PostgreSQL database")
        return True
    except Exception as e:
        logger.error(f"Error setting up PostgreSQL database: {str(e)}")
        return False

def run_migrations():
    """Run database migrations using Alembic"""
    logger.info("Running database migrations")

    try:
        base_dir = Path(__file__).resolve().parent.parent
        alembic_cfg = Config(str(base_dir / "alembic.ini"))
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error running migrations: {str(e)}")
        return False

def create_default_model_configurations():
    """Create default model configurations"""
    logger.info("Creating default model configurations")

    try:
        session = SessionLocal()
        repo = MLSettingsRepository(session)

        # Check if we already have a default config
        default_config = repo.get_default()
        if default_config:
            logger.info(f"Default model configuration already exists: {default_config.name}")
            return True

        logger.info("No default configuration found, creating new model configurations")
        
        # Create default configs for Gemini models
        try:
            repo.create_model_config(
                name="Gemini 1.5 Flash + Pro",
                primary_model="gemini-1.5-flash",
                optimizer_model="gemini-1.5-pro",
                temperature=0.2,
                max_tokens=1024,
                top_p=0.95,
                top_k=40,
                is_default=True
            )
            logger.info("Created 'Gemini 1.5 Flash + Pro' configuration")
            
            repo.create_model_config(
                name="Gemini Pro Only",
                primary_model="gemini-1.5-pro",
                optimizer_model="gemini-1.5-pro",
                temperature=0.0,
                max_tokens=2048,
                top_p=1.0,
                top_k=40,
                is_default=False
            )
            logger.info("Created 'Gemini Pro Only' configuration")
            
            # Verify configurations were created
            all_configs = session.query(ModelConfiguration).all()
            logger.info(f"Total model configurations now available: {len(all_configs)}")
            
            logger.info("Default model configurations created successfully")
            return True
        except Exception as e:
            logger.error(f"Error during model configuration creation: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"Error creating default model configurations: {str(e)}")
        return False
    finally:
        if session:
            session.close()

def check_prefect_database():
    """Check if Prefect database exists and create it if needed"""
    try:
        # Check if we're running on Replit
        is_replit = "REPL_ID" in os.environ and "REPLIT_DB_URL" in os.environ
        
        if is_replit and os.getenv("REPLIT_DB_URL"):
            # Using Replit's PostgreSQL
            main_db_url = os.getenv("DATABASE_URL")
            
            # For Replit, we need to use the same database since we can't create
            # new databases through code. We'll use schema instead.
            prefect_db_url = main_db_url
            
            # Set environment variable for Prefect
            os.environ["PREFECT_API_DATABASE_CONNECTION_URL"] = prefect_db_url
            
            # Test if we can connect
            try:
                conn = psycopg2.connect(prefect_db_url)
                cursor = conn.cursor()
                
                # Check if prefect schema exists
                cursor.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'prefect'")
                if cursor.fetchone() is None:
                    # Create schema if it doesn't exist
                    cursor.execute("CREATE SCHEMA IF NOT EXISTS prefect")
                    conn.commit()
                    logger.info("Created 'prefect' schema in Replit PostgreSQL database")
                
                cursor.close()
                conn.close()
                logger.info(f"Connected to Replit PostgreSQL for Prefect")
            except Exception as e:
                logger.warning(f"Error setting up Prefect schema: {str(e)}")
                logger.warning("Make sure your Replit PostgreSQL database is properly set up")
                return False
                
            logger.info(f"Configured Prefect to use Replit PostgreSQL database")
            return True
            
        # Standard PostgreSQL setup for non-Replit environments
        logger.info("Checking if Prefect database exists")
        
        # Get PostgreSQL URL from environment or use default
        main_db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/promptopt")
        
        # Create a prefect-specific database URL (same server, different database)
        parts = main_db_url.rsplit('/', 1)
        base_url = parts[0]
        prefect_db_url = f"{base_url}/prefect"
        
        # Create connection to PostgreSQL server for admin operations
        conn_str = base_url + "/postgres"
        conn = psycopg2.connect(conn_str)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if prefect database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'prefect'")
        if cursor.fetchone() is None:
            # Create database if it doesn't exist
            logger.info("Prefect database does not exist, creating it...")
            cursor.execute("CREATE DATABASE prefect")
            logger.info("Prefect database created successfully")
        else:
            logger.info("Prefect database already exists")
        
        cursor.close()
        conn.close()
        
        # Set environment variable for Prefect
        os.environ["PREFECT_API_DATABASE_CONNECTION_URL"] = prefect_db_url
        
        logger.info(f"Configured Prefect to use PostgreSQL database at {prefect_db_url}")
        return True
    except Exception as e:
        logger.error(f"Error configuring Prefect database: {str(e)}")
        return False

def main():
    """Main entry point for database initialization"""
    logger.info("Starting database initialization")

    # Check application database exists
    if not check_database_exists():
        logger.error("Failed to check/create application database")
        sys.exit(1)
    logger.info("✅ Application database setup complete")

    # Check Prefect database exists
    if not check_prefect_database():
        logger.error("Failed to check/create Prefect database")
        sys.exit(1)
    logger.info("✅ Prefect database setup complete")

    # Run migrations
    if not run_migrations():
        logger.error("Failed to run database migrations")
        sys.exit(1)
    logger.info("✅ Database migrations complete")

    # Create default data
    if not create_default_model_configurations():
        logger.error("Failed to create default model configurations")
        sys.exit(1)
    logger.info("✅ Default model configurations created")

    logger.info("✅ Database initialization completed successfully")
    
    # Print summary
    try:
        session = SessionLocal()
        model_count = session.query(ModelConfiguration).count()
        session.close()
        logger.info(f"Database has {model_count} model configurations available")
    except Exception as e:
        logger.warning(f"Could not count model configurations: {str(e)}")

if __name__ == "__main__":
    main()
