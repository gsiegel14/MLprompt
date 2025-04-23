#!/usr/bin/env python3
"""
Initialize database tables for the Prompt Optimization Platform
"""
import os
import sys
import logging
from datetime import datetime
import uuid

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app.database.db import engine, Base, SessionLocal
from src.app.models.database_models import (
    User, ModelConfiguration, MetricConfiguration, 
    MetaLearningConfiguration, Experiment
)
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

def create_default_user():
    """Create a default user if none exists"""
    from src.app.database.db import SessionLocal

    db = SessionLocal()
    try:
        # Check if any user exists
        user_count = db.query(User).count()

        if user_count == 0:
            logger.info("Creating default user...")
            default_user = User(
                id=str(uuid.uuid4()),
                username="admin",
                email="admin@example.com",
                api_key="ML_PLATFORM_API_KEY_" + str(uuid.uuid4()).replace("-", "")[:8]
            )
            db.add(default_user)
            db.commit()
            logger.info(f"Default user created with API key: {default_user.api_key}")
            return default_user
        else:
            logger.info("Default user already exists")
            return db.query(User).first()
    except Exception as e:
        logger.error(f"Error creating default user: {str(e)}")
        db.rollback()
        return None
    finally:
        db.close()

def create_default_settings():
    """Create default configurations"""
    from src.app.repositories.ml_settings_repository import (
        ModelConfigRepository, MetricConfigRepository, MetaLearningConfigRepository
    )

    db = SessionLocal()

    try:
        # Get or create default user
        default_user = create_default_user()
        if not default_user:
            user_id = None
        else:
            user_id = default_user.id

        # Create model config repository
        model_repo = ModelConfigRepository(db)

        # Check if default model config exists
        default_model = model_repo.get_default()

        if not default_model:
            logger.info("Creating default model configuration...")
            model_repo.create(
                name="Default Gemini Configuration",
                primary_model="gemini-1.5-flash",
                optimizer_model="gemini-1.5-pro",
                temperature=0.2,
                max_tokens=1024,
                top_p=0.95,
                top_k=40,
                is_default=True,
                user_id=user_id
            )

        # Create metric config repository
        metric_repo = MetricConfigRepository(db)

        # Create default metric configuration if none exists
        metrics = metric_repo.list_all(limit=1)
        if not metrics:
            logger.info("Creating default metric configurations...")
            metric_repo.create(
                name="Text Generation Metrics",
                metrics=["exact_match", "bleu", "rouge"],
                metric_weights={"exact_match": 0.6, "bleu": 0.2, "rouge": 0.2},
                target_threshold=0.75,
                user_id=user_id
            )

        # Create meta-learning config repository
        meta_repo = MetaLearningConfigRepository(db)

        # Create default meta-learning configuration if none exists
        meta_configs = meta_repo.list_all(limit=1)
        if not meta_configs:
            logger.info("Creating default meta-learning configuration...")
            meta_repo.create(
                name="Default XGBoost",
                model_type="xgboost",
                hyperparameters={
                    "learning_rate": 0.1,
                    "max_depth": 5,
                    "n_estimators": 100
                },
                is_active=True,
                user_id=user_id
            )

        logger.info("Default configurations created successfully!")
        return True
    except Exception as e:
        logger.error(f"Error creating default configurations: {str(e)}")
        db.rollback()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"), exist_ok=True)

    if init_database():
        create_default_settings()