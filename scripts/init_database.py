
#!/usr/bin/env python3
"""
Initialize database tables for the Prompt Optimization Platform
"""
import os
import sys
import logging

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app.factory import engine, Base
from src.app.models.ml_models import ModelConfiguration, MetricConfiguration, MetaLearningConfiguration, Experiment, User
from src.app.utils.logger import setup_logging

# Set up logging
logger = setup_logging("db_init")

def init_database():
    """Create all tables in the database"""
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully!")
        return True
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        return False

def create_default_settings():
    """Create default configurations"""
    from src.app.factory import db_session
    from src.app.utils.ml_settings_service import MLSettingsService
    
    service = MLSettingsService(db_session)
    
    try:
        # Check if default model config exists
        default_model = service.get_default_model_config()
        
        if not default_model:
            logger.info("Creating default model configuration...")
            service.create_model_config({
                "name": "Default Gemini Configuration",
                "primary_model": "gemini-1.5-flash",
                "optimizer_model": "gemini-1.5-pro",
                "temperature": 0.2,
                "max_tokens": 1024,
                "top_p": 0.95,
                "top_k": 40,
                "is_default": True
            })
        
        # Create default metric configuration if none exists
        metrics = service.get_metric_configs()
        if not metrics:
            logger.info("Creating default metric configurations...")
            service.create_metric_config({
                "name": "Text Generation Metrics",
                "metrics": ["exact_match", "bleu", "rouge"],
                "metric_weights": {"exact_match": 0.6, "bleu": 0.2, "rouge": 0.2},
                "target_threshold": 0.75
            })
        
        # Create default meta-learning configuration if none exists
        meta_configs = service.get_meta_learning_configs()
        if not meta_configs:
            logger.info("Creating default meta-learning configuration...")
            service.create_meta_learning_config({
                "name": "Default XGBoost",
                "model_type": "xgboost",
                "hyperparameters": {
                    "learning_rate": 0.1,
                    "max_depth": 5,
                    "n_estimators": 100
                },
                "is_active": True
            })
        
        logger.info("Default configurations created successfully!")
        return True
    except Exception as e:
        logger.error(f"Error creating default configurations: {str(e)}")
        return False

if __name__ == "__main__":
    if init_database():
        create_default_settings()
"""
Database initialization script for ML Prompt Optimization Platform
"""
import logging
import sqlite3
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path("data") / "ml_platform.db"

# SQL scripts for table creation
CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL,
    api_key TEXT NOT NULL UNIQUE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_MODEL_CONFIGURATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS model_configurations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    primary_model TEXT NOT NULL,
    optimizer_model TEXT NOT NULL,
    temperature REAL DEFAULT 0.0,
    max_tokens INTEGER DEFAULT 1024,
    top_p REAL DEFAULT 1.0,
    top_k INTEGER DEFAULT 40,
    is_default BOOLEAN DEFAULT FALSE,
    user_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
"""

CREATE_METRIC_CONFIGURATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS metric_configurations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    metrics TEXT NOT NULL,
    metric_weights TEXT DEFAULT '{}',
    target_threshold REAL DEFAULT 0.8,
    user_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
"""

CREATE_META_LEARNING_CONFIGURATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS meta_learning_configurations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    model_type TEXT DEFAULT 'xgboost',
    hyperparameters TEXT DEFAULT '{}',
    feature_selection TEXT DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    user_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
"""

CREATE_EXPERIMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL,
    model_config_id TEXT,
    metric_config_id TEXT,
    dataset_id TEXT,
    max_iterations INTEGER DEFAULT 5,
    strategy TEXT NOT NULL,
    iterations_completed INTEGER DEFAULT 0,
    current_best_score REAL DEFAULT 0.0,
    improvement_percentage REAL DEFAULT 0.0,
    user_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (model_config_id) REFERENCES model_configurations (id),
    FOREIGN KEY (metric_config_id) REFERENCES metric_configurations (id),
    FOREIGN KEY (user_id) REFERENCES users (id)
);
"""

def initialize_database():
    """Initialize database with required tables"""
    logger.info(f"Initializing database at {DB_PATH}")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Connect to SQLite database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Create tables
        cursor.execute(CREATE_USERS_TABLE)
        cursor.execute(CREATE_MODEL_CONFIGURATIONS_TABLE)
        cursor.execute(CREATE_METRIC_CONFIGURATIONS_TABLE)
        cursor.execute(CREATE_META_LEARNING_CONFIGURATIONS_TABLE)
        cursor.execute(CREATE_EXPERIMENTS_TABLE)
        
        # Insert default user if not exists
        cursor.execute(
            "INSERT OR IGNORE INTO users (id, username, email, api_key) VALUES (?, ?, ?, ?)",
            ("default_user", "admin", "admin@example.com", "ML_PLATFORM_API_KEY_1234")
        )
        
        # Insert default model configurations if not exists
        cursor.execute(
            """
            INSERT OR IGNORE INTO model_configurations 
            (id, name, primary_model, optimizer_model, temperature, max_tokens, is_default, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default_model_config", 
                "Standard Gemini Pro", 
                "gemini-1.5-pro", 
                "gemini-1.5-pro",
                0.2,
                1024,
                True,
                "default_user"
            )
        )
        
        # Insert default metric configuration if not exists
        cursor.execute(
            """
            INSERT OR IGNORE INTO metric_configurations
            (id, name, metrics, metric_weights, target_threshold, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "default_metric_config",
                "Standard Metrics",
                '["exact_match", "bleu", "rouge"]',
                '{"exact_match": 0.6, "bleu": 0.3, "rouge": 0.1}',
                0.85,
                "default_user"
            )
        )
        
        # Commit changes
        conn.commit()
        logger.info("Database initialization completed successfully")
        
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_database()
