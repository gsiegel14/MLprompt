
"""
API endpoints for ML settings management
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from src.app.auth import get_api_key, get_user_from_token
from src.app.utils.ml_settings_service import MLSettingsService
from src.api.models import (
    ModelConfigurationCreate,
    ModelConfigurationUpdate,
    ModelConfigurationResponse,
    MetricConfigurationCreate,
    MetricConfigurationUpdate,
    MetricConfigurationResponse,
    MetaLearningConfigurationCreate,
    MetaLearningConfigurationUpdate,
    MetaLearningConfigurationResponse
)
from src.app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

def get_db():
    """Get database session from app context"""
    # This would be replaced with your actual DB session provider
    from src.app.factory import get_db_session
    db = get_db_session()
    try:
        yield db
    finally:
        db.close()

# Model configuration endpoints
@router.post("/models", response_model=ModelConfigurationResponse, status_code=201)
async def create_model_config(
    model: ModelConfigurationCreate,
    db: Session = Depends(get_db),
    token: str = Depends(get_api_key)
):
    """Create a new model configuration"""
    user = get_user_from_token(token)
    service = MLSettingsService(db)
    
    try:
        result = service.create_model_config(model.dict(), user_id=user.get("id") if user else None)
        return result
    except Exception as e:
        logger.error(f"Error creating model config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating model configuration: {str(e)}")

@router.get("/models", response_model=List[ModelConfigurationResponse])
async def get_model_configs(
    db: Session = Depends(get_db),
    token: str = Depends(get_api_key)
):
    """Get all model configurations"""
    user = get_user_from_token(token)
    service = MLSettingsService(db)
    
    configs = service.get_model_configs(user_id=user.get("id") if user else None)
    return configs

@router.get("/models/{model_id}", response_model=ModelConfigurationResponse)
async def get_model_config(
    model_id: str,
    db: Session = Depends(get_db),
    token: str = Depends(get_api_key)
):
    """Get a specific model configuration"""
    service = MLSettingsService(db)
    config = service.get_model_config(model_id)
    
    if not config:
        raise HTTPException(status_code=404, detail="Model configuration not found")
    
    return config

@router.put("/models/{model_id}", response_model=ModelConfigurationResponse)
async def update_model_config(
    model_id: str,
    model: ModelConfigurationUpdate,
    db: Session = Depends(get_db),
    token: str = Depends(get_api_key)
):
    """Update a model configuration"""
    service = MLSettingsService(db)
    
    # Remove None values from update dict
    update_data = {k: v for k, v in model.dict().items() if v is not None}
    
    result = service.update_model_config(model_id, update_data)
    if not result:
        raise HTTPException(status_code=404, detail="Model configuration not found")
    
    return result

@router.delete("/models/{model_id}", status_code=204)
async def delete_model_config(
    model_id: str,
    db: Session = Depends(get_db),
    token: str = Depends(get_api_key)
):
    """Delete a model configuration"""
    service = MLSettingsService(db)
    
    success = service.delete_model_config(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model configuration not found")
    
    return {"success": True}

# Metric configuration endpoints
@router.post("/metrics", response_model=MetricConfigurationResponse, status_code=201)
async def create_metric_config(
    metric: MetricConfigurationCreate,
    db: Session = Depends(get_db),
    token: str = Depends(get_api_key)
):
    """Create a new metric configuration"""
    user = get_user_from_token(token)
    service = MLSettingsService(db)
    
    try:
        result = service.create_metric_config(metric.dict(), user_id=user.get("id") if user else None)
        return result
    except Exception as e:
        logger.error(f"Error creating metric config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating metric configuration: {str(e)}")

@router.get("/metrics", response_model=List[MetricConfigurationResponse])
async def get_metric_configs(
    db: Session = Depends(get_db),
    token: str = Depends(get_api_key)
):
    """Get all metric configurations"""
    user = get_user_from_token(token)
    service = MLSettingsService(db)
    
    configs = service.get_metric_configs(user_id=user.get("id") if user else None)
    return configs

@router.get("/metrics/{metric_id}", response_model=MetricConfigurationResponse)
async def get_metric_config(
    metric_id: str,
    db: Session = Depends(get_db),
    token: str = Depends(get_api_key)
):
    """Get a specific metric configuration"""
    service = MLSettingsService(db)
    config = service.get_metric_config(metric_id)
    
    if not config:
        raise HTTPException(status_code=404, detail="Metric configuration not found")
    
    return config

@router.put("/metrics/{metric_id}", response_model=MetricConfigurationResponse)
async def update_metric_config(
    metric_id: str,
    metric: MetricConfigurationUpdate,
    db: Session = Depends(get_db),
    token: str = Depends(get_api_key)
):
    """Update a metric configuration"""
    service = MLSettingsService(db)
    
    # Remove None values from update dict
    update_data = {k: v for k, v in metric.dict().items() if v is not None}
    
    result = service.update_metric_config(metric_id, update_data)
    if not result:
        raise HTTPException(status_code=404, detail="Metric configuration not found")
    
    return result

@router.delete("/metrics/{metric_id}", status_code=204)
async def delete_metric_config(
    metric_id: str,
    db: Session = Depends(get_db),
    token: str = Depends(get_api_key)
):
    """Delete a metric configuration"""
    service = MLSettingsService(db)
    
    success = service.delete_metric_config(metric_id)
    if not success:
        raise HTTPException(status_code=404, detail="Metric configuration not found")
    
    return {"success": True}

# Meta-learning configuration endpoints
@router.post("/meta-learning", response_model=MetaLearningConfigurationResponse, status_code=201)
async def create_meta_learning_config(
    config: MetaLearningConfigurationCreate,
    db: Session = Depends(get_db),
    token: str = Depends(get_api_key)
):
    """Create a new meta-learning configuration"""
    user = get_user_from_token(token)
    service = MLSettingsService(db)
    
    try:
        result = service.create_meta_learning_config(config.dict(), user_id=user.get("id") if user else None)
        return result
    except Exception as e:
        logger.error(f"Error creating meta-learning config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating meta-learning configuration: {str(e)}")

@router.get("/meta-learning", response_model=List[MetaLearningConfigurationResponse])
async def get_meta_learning_configs(
    db: Session = Depends(get_db),
    token: str = Depends(get_api_key)
):
    """Get all meta-learning configurations"""
    user = get_user_from_token(token)
    service = MLSettingsService(db)
    
    configs = service.get_meta_learning_configs(user_id=user.get("id") if user else None)
    return configs
