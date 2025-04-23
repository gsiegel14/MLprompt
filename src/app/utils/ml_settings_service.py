
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
