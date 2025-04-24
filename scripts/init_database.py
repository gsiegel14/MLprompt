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
    db_url = os.getenv("DATABASE_URL", "postgresql://promptopt:devpassword@localhost:5432/promptopt")

    # Parse the URL to get database name and connection string
    parts = db_url.split("/")
    db_name = parts[-1]
    connection_str = "/".join(parts[:-1]) + "/postgres"

    logger.info(f"Checking if database '{db_name}' exists")

    try:
        # Connect to postgres database to create our database if needed
        conn = psycopg2.connect(connection_str)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()

        if not exists:
            logger.info(f"Creating database '{db_name}'")
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            logger.info(f"Database '{db_name}' created successfully")
        else:
            logger.info(f"Database '{db_name}' already exists")

        cursor.close()
        conn.close()

        return True
    except Exception as e:
        logger.error(f"Error checking/creating database: {str(e)}")
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

        # Create default configs for Gemini models
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

        logger.info("Default model configurations created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating default model configurations: {str(e)}")
        return False
    finally:
        session.close()

def main():
    """Main entry point for database initialization"""
    logger.info("Starting database initialization")

    # Check database exists
    if not check_database_exists():
        logger.error("Failed to check/create database")
        sys.exit(1)

    # Run migrations
    if not run_migrations():
        logger.error("Failed to run database migrations")
        sys.exit(1)

    # Create default data
    if not create_default_model_configurations():
        logger.error("Failed to create default model configurations")
        sys.exit(1)

    logger.info("Database initialization completed successfully")

if __name__ == "__main__":
    main()