
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from src.app.models.database_models import ModelConfiguration
import uuid

class MLSettingsRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create_model_config(self, name: str, primary_model: str, optimizer_model: str,
                         temperature: float = 0.0, max_tokens: int = 1024,
                         top_p: float = 1.0, top_k: int = 40,
                         is_default: bool = False) -> ModelConfiguration:
        """Create a new model configuration"""
        # If setting as default, unset any existing defaults
        if is_default:
            self._unset_all_defaults()
        
        model_config = ModelConfiguration(
            name=name,
            primary_model=primary_model,
            optimizer_model=optimizer_model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            is_default=is_default
        )
        
        self.db.add(model_config)
        self.db.commit()
        self.db.refresh(model_config)
        return model_config
    
    def _unset_all_defaults(self):
        """Unset all default model configurations"""
        self.db.query(ModelConfiguration).filter(ModelConfiguration.is_default == True).update(
            {"is_default": False}, synchronize_session=False
        )
        self.db.commit()
    
    def get_by_id(self, config_id: str) -> Optional[ModelConfiguration]:
        """Get model configuration by ID"""
        return self.db.query(ModelConfiguration).filter(ModelConfiguration.id == uuid.UUID(config_id)).first()
    
    def get_default(self) -> Optional[ModelConfiguration]:
        """Get the default model configuration"""
        return self.db.query(ModelConfiguration).filter(ModelConfiguration.is_default == True).first()
    
    def set_as_default(self, config_id: str) -> Optional[ModelConfiguration]:
        """Set a model configuration as the default"""
        self._unset_all_defaults()
        
        config = self.get_by_id(config_id)
        if config:
            config.is_default = True
            self.db.commit()
            self.db.refresh(config)
            return config
        return None
    
    def list_model_configs(self) -> List[ModelConfiguration]:
        """List all model configurations"""
        return self.db.query(ModelConfiguration).order_by(ModelConfiguration.name).all()
    
    def update(self, config_id: str, data: Dict[str, Any]) -> Optional[ModelConfiguration]:
        """Update a model configuration"""
        config = self.get_by_id(config_id)
        if not config:
            return None
            
        # Update fields
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Handle is_default specially
        if 'is_default' in data and data['is_default']:
            self._unset_all_defaults()
            config.is_default = True
            
        self.db.commit()
        self.db.refresh(config)
        return config
    
    def delete(self, config_id: str) -> bool:
        """Delete a model configuration"""
        config = self.get_by_id(config_id)
        if config:
            self.db.delete(config)
            self.db.commit()
            return True
        return False
