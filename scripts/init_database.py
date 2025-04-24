
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
    # For Replit, we'll use SQLite as a fallback if PostgreSQL is not configured
    db_url = os.getenv("DATABASE_URL")
    
    # If no DATABASE_URL is provided, default to SQLite
    if not db_url:
        db_dir = Path(__file__).parent.parent / "data" / "db"
        db_dir.mkdir(exist_ok=True, parents=True)
        db_path = db_dir / "promptopt.db"
        db_url = f"sqlite:///{db_path}"
        os.environ["DATABASE_URL"] = db_url
        logger.info(f"No PostgreSQL configured, using SQLite at {db_path}")
        return True
        
    # If PostgreSQL URL is provided, attempt to connect
    if db_url.startswith("postgresql"):
        try:
            # For Replit PostgreSQL, we don't need to create the database
            # Just verify connection works
            conn = psycopg2.connect(db_url)
            conn.close()
            logger.info("Successfully connected to PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {str(e)}")
            logger.warning("Falling back to SQLite database")
            # Fall back to SQLite
            db_dir = Path(__file__).parent.parent / "data" / "db"
            db_dir.mkdir(exist_ok=True, parents=True)
            db_path = db_dir / "promptopt.db"
            db_url = f"sqlite:///{db_path}"
            os.environ["DATABASE_URL"] = db_url
            logger.info(f"Using SQLite at {db_path}")
            return True
    
    # Non-PostgreSQL URL was provided (like SQLite)
    return True

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
    # For Replit environment, we'll configure Prefect to use SQLite instead of PostgreSQL
    try:
        # Create directory for Prefect SQLite database if it doesn't exist
        prefect_db_dir = Path(__file__).parent.parent / "data" / "prefect"
        prefect_db_dir.mkdir(exist_ok=True, parents=True)
        prefect_db_path = prefect_db_dir / "prefect.db"
        
        # Set the Prefect database URL to use SQLite
        prefect_db_url = f"sqlite:///{prefect_db_path}"
        os.environ["PREFECT_API_DATABASE_CONNECTION_URL"] = prefect_db_url
        
        logger.info(f"Configured Prefect to use SQLite database at {prefect_db_path}")
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
