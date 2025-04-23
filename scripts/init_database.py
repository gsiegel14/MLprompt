
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
