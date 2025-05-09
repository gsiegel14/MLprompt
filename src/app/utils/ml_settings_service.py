
"""
Service for managing ML settings
"""
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime

from src.app.models.ml_models import (
    ModelConfiguration, 
    MetricConfiguration, 
    MetaLearningConfiguration,
    Experiment
)

class MLSettingsService:
    def __init__(self, db: Session):
        self.db = db
    
    # Model Configuration Methods
    def create_model_config(self, config_data: Dict[str, Any], user_id: Optional[str] = None) -> ModelConfiguration:
        """Create a new model configuration"""
        model_config = ModelConfiguration(
            name=config_data.get("name"),
            primary_model=config_data.get("primary_model"),
            optimizer_model=config_data.get("optimizer_model"),
            temperature=config_data.get("temperature", 0.0),
            max_tokens=config_data.get("max_tokens", 1024),
            top_p=config_data.get("top_p", 1.0),
            top_k=config_data.get("top_k", 40),
            is_default=config_data.get("is_default", False),
            user_id=user_id
        )
        
        # If this is set as default, unset other defaults
        if model_config.is_default:
            existing_defaults = self.db.query(ModelConfiguration).filter_by(is_default=True).all()
            for default in existing_defaults:
                default.is_default = False
        
        self.db.add(model_config)
        self.db.commit()
        self.db.refresh(model_config)
        return model_config
    
    def get_model_configs(self, user_id: Optional[str] = None) -> List[ModelConfiguration]:
        """Get all model configurations for a user"""
        query = self.db.query(ModelConfiguration)
        if user_id:
            query = query.filter_by(user_id=user_id)
        return query.all()
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfiguration]:
        """Get a specific model configuration"""
        return self.db.query(ModelConfiguration).filter_by(id=model_id).first()
    
    def get_default_model_config(self) -> Optional[ModelConfiguration]:
        """Get the default model configuration"""
        return self.db.query(ModelConfiguration).filter_by(is_default=True).first()
    
    def update_model_config(self, model_id: str, config_data: Dict[str, Any]) -> Optional[ModelConfiguration]:
        """Update a model configuration"""
        model_config = self.db.query(ModelConfiguration).filter_by(id=model_id).first()
        if not model_config:
            return None
        
        # Update fields
        for key, value in config_data.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
        
        # Handle default setting
        if config_data.get("is_default", False):
            existing_defaults = self.db.query(ModelConfiguration).filter(
                ModelConfiguration.id != model_id, 
                ModelConfiguration.is_default == True
            ).all()
            for default in existing_defaults:
                default.is_default = False
        
        self.db.commit()
        self.db.refresh(model_config)
        return model_config
    
    def delete_model_config(self, model_id: str) -> bool:
        """Delete a model configuration"""
        model_config = self.db.query(ModelConfiguration).filter_by(id=model_id).first()
        if not model_config:
            return False
        
        self.db.delete(model_config)
        self.db.commit()
        return True
    
    # Metric Configuration Methods
    def create_metric_config(self, config_data: Dict[str, Any], user_id: Optional[str] = None) -> MetricConfiguration:
        """Create a new metric configuration"""
        metric_config = MetricConfiguration(
            name=config_data.get("name"),
            metrics=config_data.get("metrics", []),
            metric_weights=config_data.get("metric_weights", {}),
            target_threshold=config_data.get("target_threshold", 0.8),
            user_id=user_id
        )
        
        self.db.add(metric_config)
        self.db.commit()
        self.db.refresh(metric_config)
        return metric_config
    
    def get_metric_configs(self, user_id: Optional[str] = None) -> List[MetricConfiguration]:
        """Get all metric configurations for a user"""
        query = self.db.query(MetricConfiguration)
        if user_id:
            query = query.filter_by(user_id=user_id)
        return query.all()
    
    def get_metric_config(self, metric_id: str) -> Optional[MetricConfiguration]:
        """Get a specific metric configuration"""
        return self.db.query(MetricConfiguration).filter_by(id=metric_id).first()
    
    def update_metric_config(self, metric_id: str, config_data: Dict[str, Any]) -> Optional[MetricConfiguration]:
        """Update a metric configuration"""
        metric_config = self.db.query(MetricConfiguration).filter_by(id=metric_id).first()
        if not metric_config:
            return None
        
        # Update fields
        for key, value in config_data.items():
            if hasattr(metric_config, key):
                setattr(metric_config, key, value)
        
        self.db.commit()
        self.db.refresh(metric_config)
        return metric_config
    
    def delete_metric_config(self, metric_id: str) -> bool:
        """Delete a metric configuration"""
        metric_config = self.db.query(MetricConfiguration).filter_by(id=metric_id).first()
        if not metric_config:
            return False
        
        self.db.delete(metric_config)
        self.db.commit()
        return True
    
    # Meta-Learning Configuration Methods
    def create_meta_learning_config(self, config_data: Dict[str, Any], user_id: Optional[str] = None) -> MetaLearningConfiguration:
        """Create a new meta-learning configuration"""
        meta_config = MetaLearningConfiguration(
            name=config_data.get("name"),
            model_type=config_data.get("model_type", "xgboost"),
            hyperparameters=config_data.get("hyperparameters", {}),
            feature_selection=config_data.get("feature_selection", {}),
            is_active=config_data.get("is_active", True),
            user_id=user_id
        )
        
        self.db.add(meta_config)
        self.db.commit()
        self.db.refresh(meta_config)
        return meta_config
    
    def get_meta_learning_configs(self, user_id: Optional[str] = None) -> List[MetaLearningConfiguration]:
        """Get all meta-learning configurations for a user"""
        query = self.db.query(MetaLearningConfiguration)
        if user_id:
            query = query.filter_by(user_id=user_id)
        return query.all()
    
    def get_meta_learning_config(self, config_id: str) -> Optional[MetaLearningConfiguration]:
        """Get a specific meta-learning configuration"""
        return self.db.query(MetaLearningConfiguration).filter_by(id=config_id).first()
    
    def update_meta_learning_config(self, config_id: str, config_data: Dict[str, Any]) -> Optional[MetaLearningConfiguration]:
        """Update a meta-learning configuration"""
        meta_config = self.db.query(MetaLearningConfiguration).filter_by(id=config_id).first()
        if not meta_config:
            return None
        
        # Update fields
        for key, value in config_data.items():
            if hasattr(meta_config, key):
                setattr(meta_config, key, value)
        
        self.db.commit()
        self.db.refresh(meta_config)
        return meta_config
    
    def delete_meta_learning_config(self, config_id: str) -> bool:
        """Delete a meta-learning configuration"""
        meta_config = self.db.query(MetaLearningConfiguration).filter_by(id=config_id).first()
        if not meta_config:
            return False
        
        self.db.delete(meta_config)
        self.db.commit()
        return True
    
    # Experiment Methods
    def create_experiment(self, 
                         name: str, 
                         model_config_id: Optional[str] = None, 
                         user_id: Optional[str] = None) -> Experiment:
        """Create a new experiment record"""
        experiment = Experiment(
            name=name,
            model_config_id=model_config_id,
            start_time=datetime.now().isoformat(),
            status="pending",
            user_id=user_id
        )
        
        self.db.add(experiment)
        self.db.commit()
        self.db.refresh(experiment)
        return experiment
    
    def update_experiment_status(self, 
                                experiment_id: str, 
                                status: str, 
                                metrics: Optional[Dict[str, Any]] = None,
                                iterations: Optional[int] = None,
                                best_iteration: Optional[int] = None) -> Optional[Experiment]:
        """Update experiment status and results"""
        experiment = self.db.query(Experiment).filter_by(id=experiment_id).first()
        if not experiment:
            return None
        
        experiment.status = status
        
        if status == "completed":
            experiment.end_time = datetime.now().isoformat()
        
        if metrics:
            experiment.metrics = metrics
        
        if iterations is not None:
            experiment.iterations = iterations
        
        if best_iteration is not None:
            experiment.best_iteration = best_iteration
        
        self.db.commit()
        self.db.refresh(experiment)
        return experiment
    
    def get_experiments(self, user_id: Optional[str] = None, limit: int = 100) -> List[Experiment]:
        """Get all experiments for a user, ordered by most recent first"""
        query = self.db.query(Experiment)
        if user_id:
            query = query.filter_by(user_id=user_id)
        return query.order_by(Experiment.start_time.desc()).limit(limit).all()
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get a specific experiment"""
        return self.db.query(Experiment).filter_by(id=experiment_id).first()
"""
Service functions for ML settings management
"""
import uuid
from typing import Dict, List, Any, Optional

# This is a temporary in-memory store - would be replaced with database in production
_model_configs = {}
_metric_configs = {}
_meta_learning_configs = {}

def get_model_configurations(user_id: str) -> List[Dict[str, Any]]:
    """Get all model configurations for a user"""
    return [cfg for cfg in _model_configs.values() if cfg.get("user_id") == user_id]

def get_model_configuration(config_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific model configuration by ID"""
    config = _model_configs.get(config_id)
    if config and config.get("user_id") == user_id:
        return config
    return None

def create_model_configuration(config_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Create a new model configuration"""
    config_id = str(uuid.uuid4())
    config = {
        "id": config_id,
        "user_id": user_id,
        **config_data
    }
    _model_configs[config_id] = config
    return config

def update_model_configuration(config_id: str, config_data: Dict[str, Any], user_id: str) -> Optional[Dict[str, Any]]:
    """Update an existing model configuration"""
    if config_id in _model_configs and _model_configs[config_id].get("user_id") == user_id:
        _model_configs[config_id].update(config_data)
        return _model_configs[config_id]
    return None

def delete_model_configuration(config_id: str, user_id: str) -> bool:
    """Delete a model configuration"""
    if config_id in _model_configs and _model_configs[config_id].get("user_id") == user_id:
        del _model_configs[config_id]
        return True
    return False

def get_metric_configurations(user_id: str) -> List[Dict[str, Any]]:
    """Get all metric configurations for a user"""
    return [cfg for cfg in _metric_configs.values() if cfg.get("user_id") == user_id]

def create_metric_configuration(config_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Create a new metric configuration"""
    config_id = str(uuid.uuid4())
    config = {
        "id": config_id,
        "user_id": user_id,
        **config_data
    }
    _metric_configs[config_id] = config
    return config

def get_meta_learning_configurations(user_id: str) -> List[Dict[str, Any]]:
    """Get all meta-learning configurations for a user"""
    return [cfg for cfg in _meta_learning_configs.values() if cfg.get("user_id") == user_id]

def create_meta_learning_configuration(config_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Create a new meta-learning configuration"""
    config_id = str(uuid.uuid4())
    config = {
        "id": config_id,
        "user_id": user_id,
        **config_data
    }
    _meta_learning_configs[config_id] = config
    return config

# Initialize with some default configurations for development
def initialize_default_configs():
    """Initialize default configurations for development"""
    # Default user
    default_user_id = "default_user"
    
    # Default model configurations
    if not _model_configs:
        create_model_configuration({
            "name": "Standard Gemini Pro",
            "primary_model": "gemini-1.5-pro",
            "optimizer_model": "gemini-1.5-pro",
            "temperature": 0.2,
            "max_tokens": 1024,
            "is_default": True
        }, default_user_id)
        
        create_model_configuration({
            "name": "Medical Expert",
            "primary_model": "gemini-1.5-pro",
            "optimizer_model": "gemini-1.5-pro",
            "temperature": 0.1,
            "max_tokens": 2048,
            "top_p": 0.95
        }, default_user_id)
    
    # Default metric configurations
    if not _metric_configs:
        create_metric_configuration({
            "name": "Standard Metrics",
            "metrics": ["exact_match", "bleu", "rouge"],
            "metric_weights": {
                "exact_match": 0.6,
                "bleu": 0.3,
                "rouge": 0.1
            },
            "target_threshold": 0.85
        }, default_user_id)
        
        create_metric_configuration({
            "name": "Medical Diagnosis Metrics",
            "metrics": ["exact_match", "token_overlap", "clinical_accuracy"],
            "metric_weights": {
                "exact_match": 0.4,
                "token_overlap": 0.3,
                "clinical_accuracy": 0.3
            },
            "target_threshold": 0.90
        }, default_user_id)
    
    # Default meta-learning configurations
    if not _meta_learning_configs:
        create_meta_learning_configuration({
            "name": "XGBoost Default",
            "model_type": "xgboost",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1
            },
            "feature_selection": {
                "primary_metrics": ["exact_match", "bleu"],
                "feature_importance_threshold": 0.05
            },
            "is_active": True
        }, default_user_id)

# Initialize default configurations
initialize_default_configs()
