
"""
Repository for ML settings database operations
"""
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from src.app.models.database_models import ModelConfiguration, MetricConfiguration, MetaLearningConfiguration

class ModelConfigRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, name: str, primary_model: str, optimizer_model: str, 
               temperature: float = 0.0, max_tokens: int = 1024, top_p: float = 1.0,
               top_k: int = 40, is_default: bool = False, user_id: Optional[str] = None) -> ModelConfiguration:
        """Create a new model configuration"""
        if is_default:
            # If this is a default config, unset any other defaults
            self._unset_defaults(user_id)
            
        model_config = ModelConfiguration(
            name=name,
            primary_model=primary_model,
            optimizer_model=optimizer_model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            is_default=is_default,
            user_id=user_id
        )
        
        self.db.add(model_config)
        self.db.commit()
        self.db.refresh(model_config)
        return model_config
    
    def _unset_defaults(self, user_id: Optional[str] = None):
        """Unset default flag for all model configurations"""
        query = self.db.query(ModelConfiguration).filter(ModelConfiguration.is_default == True)
        if user_id:
            query = query.filter(ModelConfiguration.user_id == user_id)
            
        for config in query.all():
            config.is_default = False
            
        self.db.commit()
    
    def get_by_id(self, config_id: str) -> Optional[ModelConfiguration]:
        """Get model configuration by ID"""
        return self.db.query(ModelConfiguration).filter(ModelConfiguration.id == config_id).first()
    
    def get_default(self, user_id: Optional[str] = None) -> Optional[ModelConfiguration]:
        """Get the default model configuration"""
        query = self.db.query(ModelConfiguration).filter(ModelConfiguration.is_default == True)
        if user_id:
            query = query.filter(ModelConfiguration.user_id == user_id)
        return query.first()
    
    def list_all(self, limit: int = 100, skip: int = 0, user_id: Optional[str] = None) -> List[ModelConfiguration]:
        """List all model configurations"""
        query = self.db.query(ModelConfiguration)
        if user_id:
            query = query.filter(ModelConfiguration.user_id == user_id)
        return query.order_by(ModelConfiguration.created_at.desc()).offset(skip).limit(limit).all()
    
    def update(self, config_id: str, **kwargs) -> Optional[ModelConfiguration]:
        """Update a model configuration"""
        config = self.get_by_id(config_id)
        if not config:
            return None
            
        # Handle default flag specially
        if 'is_default' in kwargs and kwargs['is_default'] and not config.is_default:
            self._unset_defaults(config.user_id)
            
        # Update fields
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        self.db.commit()
        self.db.refresh(config)
        return config
    
    def delete(self, config_id: str) -> bool:
        """Delete a model configuration"""
        config = self.get_by_id(config_id)
        if not config:
            return False
            
        self.db.delete(config)
        self.db.commit()
        return True

class MetricConfigRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, name: str, metrics: List[str], metric_weights: Dict[str, float] = None, 
               target_threshold: float = 0.8, user_id: Optional[str] = None) -> MetricConfiguration:
        """Create a new metric configuration"""
        metric_config = MetricConfiguration(
            name=name,
            metrics=metrics,
            metric_weights=metric_weights or {},
            target_threshold=target_threshold,
            user_id=user_id
        )
        
        self.db.add(metric_config)
        self.db.commit()
        self.db.refresh(metric_config)
        return metric_config
    
    def get_by_id(self, config_id: str) -> Optional[MetricConfiguration]:
        """Get metric configuration by ID"""
        return self.db.query(MetricConfiguration).filter(MetricConfiguration.id == config_id).first()
    
    def list_all(self, limit: int = 100, skip: int = 0, user_id: Optional[str] = None) -> List[MetricConfiguration]:
        """List all metric configurations"""
        query = self.db.query(MetricConfiguration)
        if user_id:
            query = query.filter(MetricConfiguration.user_id == user_id)
        return query.order_by(MetricConfiguration.created_at.desc()).offset(skip).limit(limit).all()
    
    def update(self, config_id: str, **kwargs) -> Optional[MetricConfiguration]:
        """Update a metric configuration"""
        config = self.get_by_id(config_id)
        if not config:
            return None
            
        # Update fields
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        self.db.commit()
        self.db.refresh(config)
        return config
    
    def delete(self, config_id: str) -> bool:
        """Delete a metric configuration"""
        config = self.get_by_id(config_id)
        if not config:
            return False
            
        self.db.delete(config)
        self.db.commit()
        return True

class MetaLearningConfigRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, name: str, model_type: str = "xgboost", hyperparameters: Dict[str, Any] = None,
               feature_selection: Dict[str, Any] = None, is_active: bool = True, 
               user_id: Optional[str] = None) -> MetaLearningConfiguration:
        """Create a new meta-learning configuration"""
        meta_config = MetaLearningConfiguration(
            name=name,
            model_type=model_type,
            hyperparameters=hyperparameters or {},
            feature_selection=feature_selection or {},
            is_active=is_active,
            user_id=user_id
        )
        
        self.db.add(meta_config)
        self.db.commit()
        self.db.refresh(meta_config)
        return meta_config
    
    def get_by_id(self, config_id: str) -> Optional[MetaLearningConfiguration]:
        """Get meta-learning configuration by ID"""
        return self.db.query(MetaLearningConfiguration).filter(MetaLearningConfiguration.id == config_id).first()
    
    def get_active(self, user_id: Optional[str] = None) -> List[MetaLearningConfiguration]:
        """Get active meta-learning configurations"""
        query = self.db.query(MetaLearningConfiguration).filter(MetaLearningConfiguration.is_active == True)
        if user_id:
            query = query.filter(MetaLearningConfiguration.user_id == user_id)
        return query.all()
    
    def list_all(self, limit: int = 100, skip: int = 0, user_id: Optional[str] = None) -> List[MetaLearningConfiguration]:
        """List all meta-learning configurations"""
        query = self.db.query(MetaLearningConfiguration)
        if user_id:
            query = query.filter(MetaLearningConfiguration.user_id == user_id)
        return query.order_by(MetaLearningConfiguration.created_at.desc()).offset(skip).limit(limit).all()
    
    def update(self, config_id: str, **kwargs) -> Optional[MetaLearningConfiguration]:
        """Update a meta-learning configuration"""
        config = self.get_by_id(config_id)
        if not config:
            return None
            
        # Update fields
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        self.db.commit()
        self.db.refresh(config)
        return config
    
    def delete(self, config_id: str) -> bool:
        """Delete a meta-learning configuration"""
        config = self.get_by_id(config_id)
        if not config:
            return False
            
        self.db.delete(config)
        self.db.commit()
        return True
