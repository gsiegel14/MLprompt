
from sqlalchemy.orm import Session
from src.app.models.ml_models import ModelConfiguration, MetricConfiguration, MetaLearningConfiguration
from typing import Dict, Any, List, Optional
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class MLSettingsService:
    def __init__(self, db: Session):
        self.db = db
    
    # Model Configuration Methods
    def get_model_configurations(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all model configurations, optionally filtered by user_id"""
        query = self.db.query(ModelConfiguration)
        if user_id:
            query = query.filter(ModelConfiguration.user_id == user_id)
        
        configs = query.all()
        return [self._model_to_dict(config) for config in configs]
    
    def get_model_configuration(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Get a model configuration by ID"""
        config = self.db.query(ModelConfiguration).filter(
            ModelConfiguration.id == config_id
        ).first()
        
        if not config:
            return None
        
        return self._model_to_dict(config)
    
    def create_model_configuration(self, config_data: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new model configuration"""
        config_id = str(uuid.uuid4())
        
        config = ModelConfiguration(
            id=config_id,
            name=config_data.get("name", "Default Configuration"),
            primary_model=config_data.get("primary_model", "gemini-1.5-flash"),
            optimizer_model=config_data.get("optimizer_model", "gemini-1.5-pro"),
            temperature=float(config_data.get("temperature", 0.0)),
            max_tokens=int(config_data.get("max_tokens", 1024)),
            top_p=float(config_data.get("top_p", 1.0)),
            top_k=int(config_data.get("top_k", 40)),
            is_default=bool(config_data.get("is_default", False)),
            user_id=user_id
        )
        
        if config.is_default:
            # If this config is set as default, unset any existing defaults
            existing_defaults = self.db.query(ModelConfiguration).filter(
                ModelConfiguration.is_default == True
            )
            if user_id:
                existing_defaults = existing_defaults.filter(ModelConfiguration.user_id == user_id)
            
            existing_defaults = existing_defaults.all()
            for default_config in existing_defaults:
                default_config.is_default = False
        
        try:
            self.db.add(config)
            self.db.commit()
            self.db.refresh(config)
            return self._model_to_dict(config)
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating model configuration: {str(e)}")
            raise
    
    def update_model_configuration(self, config_id: str, config_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a model configuration"""
        config = self.db.query(ModelConfiguration).filter(
            ModelConfiguration.id == config_id
        ).first()
        
        if not config:
            return None
            
        for key, value in config_data.items():
            if hasattr(config, key):
                if key == "temperature" or key == "top_p":
                    value = float(value)
                elif key == "max_tokens" or key == "top_k":
                    value = int(value)
                elif key == "is_default":
                    value = bool(value)
                
                setattr(config, key, value)
        
        if config_data.get("is_default", False):
            # If this config is being set as default, unset any existing defaults
            existing_defaults = self.db.query(ModelConfiguration).filter(
                ModelConfiguration.is_default == True,
                ModelConfiguration.id != config_id
            )
            if config.user_id:
                existing_defaults = existing_defaults.filter(ModelConfiguration.user_id == config.user_id)
            
            existing_defaults = existing_defaults.all()
            for default_config in existing_defaults:
                default_config.is_default = False
        
        try:
            self.db.commit()
            self.db.refresh(config)
            return self._model_to_dict(config)
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating model configuration: {str(e)}")
            raise
    
    def delete_model_configuration(self, config_id: str) -> bool:
        """Delete a model configuration"""
        config = self.db.query(ModelConfiguration).filter(
            ModelConfiguration.id == config_id
        ).first()
        
        if not config:
            return False
        
        try:
            self.db.delete(config)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting model configuration: {str(e)}")
            raise
    
    # Metric Configuration Methods
    def get_metric_configurations(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all metric configurations, optionally filtered by user_id"""
        query = self.db.query(MetricConfiguration)
        if user_id:
            query = query.filter(MetricConfiguration.user_id == user_id)
        
        configs = query.all()
        return [self._model_to_dict(config) for config in configs]
    
    def get_metric_configuration(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Get a metric configuration by ID"""
        config = self.db.query(MetricConfiguration).filter(
            MetricConfiguration.id == config_id
        ).first()
        
        if not config:
            return None
        
        return self._model_to_dict(config)
    
    def create_metric_configuration(self, config_data: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new metric configuration"""
        config_id = str(uuid.uuid4())
        
        config = MetricConfiguration(
            id=config_id,
            name=config_data.get("name", "Default Metrics"),
            metrics=config_data.get("metrics", ["exact_match"]),
            metric_weights=config_data.get("metric_weights", {}),
            target_threshold=float(config_data.get("target_threshold", 0.8)),
            user_id=user_id
        )
        
        try:
            self.db.add(config)
            self.db.commit()
            self.db.refresh(config)
            return self._model_to_dict(config)
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating metric configuration: {str(e)}")
            raise
    
    # Meta-Learning Configuration Methods
    def get_meta_learning_configurations(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all meta-learning configurations, optionally filtered by user_id"""
        query = self.db.query(MetaLearningConfiguration)
        if user_id:
            query = query.filter(MetaLearningConfiguration.user_id == user_id)
        
        configs = query.all()
        return [self._model_to_dict(config) for config in configs]
    
    def get_meta_learning_configuration(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Get a meta-learning configuration by ID"""
        config = self.db.query(MetaLearningConfiguration).filter(
            MetaLearningConfiguration.id == config_id
        ).first()
        
        if not config:
            return None
        
        return self._model_to_dict(config)
    
    def create_meta_learning_configuration(self, config_data: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new meta-learning configuration"""
        config_id = str(uuid.uuid4())
        
        config = MetaLearningConfiguration(
            id=config_id,
            name=config_data.get("name", "Default Meta-Learning"),
            model_type=config_data.get("model_type", "xgboost"),
            hyperparameters=config_data.get("hyperparameters", {}),
            feature_selection=config_data.get("feature_selection", {}),
            is_active=bool(config_data.get("is_active", True)),
            user_id=user_id
        )
        
        try:
            self.db.add(config)
            self.db.commit()
            self.db.refresh(config)
            return self._model_to_dict(config)
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating meta-learning configuration: {str(e)}")
            raise
    
    def update_meta_learning_performance(self, config_id: str, performance: float, model_path: str) -> bool:
        """Update meta-learning performance after training"""
        config = self.db.query(MetaLearningConfiguration).filter(
            MetaLearningConfiguration.id == config_id
        ).first()
        
        if not config:
            return False
        
        try:
            config.performance = performance
            config.model_path = model_path
            config.last_trained = datetime.now()
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating meta-learning performance: {str(e)}")
            raise
    
    def _model_to_dict(self, model) -> Dict[str, Any]:
        """Convert SQLAlchemy model instance to dictionary"""
        result = {}
        for column in model.__table__.columns:
            value = getattr(model, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            result[column.name] = value
        return result
