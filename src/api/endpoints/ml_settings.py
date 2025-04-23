
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
"""
ML Settings API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

from src.app.utils.ml_settings_service import (
    get_model_configurations,
    get_model_configuration,
    create_model_configuration,
    update_model_configuration,
    delete_model_configuration,
    get_metric_configurations,
    create_metric_configuration,
    get_meta_learning_configurations
)
from src.app.auth import get_api_key

router = APIRouter()

class ModelConfigBase(BaseModel):
    name: str
    primary_model: str
    optimizer_model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    top_k: int = 40
    is_default: bool = False

class ModelConfigCreate(ModelConfigBase):
    pass

class ModelConfigResponse(ModelConfigBase):
    id: str
    user_id: Optional[str] = None

class MetricConfigBase(BaseModel):
    name: str
    metrics: List[str]
    metric_weights: Dict[str, float] = {}
    target_threshold: float = 0.8

class MetricConfigCreate(MetricConfigBase):
    pass

class MetricConfigResponse(MetricConfigBase):
    id: str
    user_id: Optional[str] = None

class MetaLearningConfigBase(BaseModel):
    name: str
    model_type: str = "xgboost"
    hyperparameters: Dict[str, Any] = {}
    feature_selection: Dict[str, Any] = {}
    is_active: bool = True

class MetaLearningConfigCreate(MetaLearningConfigBase):
    pass

class MetaLearningConfigResponse(MetaLearningConfigBase):
    id: str
    user_id: Optional[str] = None

# Model Configuration Endpoints
@router.get("/models", response_model=List[ModelConfigResponse])
async def list_model_configurations(api_key: str = Depends(get_api_key)):
    """Get all model configurations for the current user"""
    return get_model_configurations(api_key)

@router.get("/models/{model_id}", response_model=ModelConfigResponse)
async def get_model_configuration_by_id(model_id: str, api_key: str = Depends(get_api_key)):
    """Get a specific model configuration by ID"""
    model_config = get_model_configuration(model_id, api_key)
    if not model_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model configuration with ID {model_id} not found"
        )
    return model_config

@router.post("/models", response_model=ModelConfigResponse, status_code=status.HTTP_201_CREATED)
async def create_new_model_configuration(
    model_config: ModelConfigCreate, 
    api_key: str = Depends(get_api_key)
):
    """Create a new model configuration"""
    return create_model_configuration(model_config.dict(), api_key)

@router.put("/models/{model_id}", response_model=ModelConfigResponse)
async def update_model_configuration_by_id(
    model_id: str, 
    model_config: ModelConfigBase,
    api_key: str = Depends(get_api_key)
):
    """Update an existing model configuration"""
    updated_config = update_model_configuration(model_id, model_config.dict(), api_key)
    if not updated_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model configuration with ID {model_id} not found"
        )
    return updated_config

@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_configuration_by_id(model_id: str, api_key: str = Depends(get_api_key)):
    """Delete a model configuration"""
    success = delete_model_configuration(model_id, api_key)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model configuration with ID {model_id} not found"
        )
    return {"detail": "Model configuration deleted successfully"}

# Metric Configuration Endpoints
@router.get("/metrics", response_model=List[MetricConfigResponse])
async def list_metric_configurations(api_key: str = Depends(get_api_key)):
    """Get all metric configurations for the current user"""
    return get_metric_configurations(api_key)

@router.post("/metrics", response_model=MetricConfigResponse, status_code=status.HTTP_201_CREATED)
async def create_new_metric_configuration(
    metric_config: MetricConfigCreate, 
    api_key: str = Depends(get_api_key)
):
    """Create a new metric configuration"""
    return create_metric_configuration(metric_config.dict(), api_key)

# Meta-Learning Configuration Endpoints
@router.get("/meta-learning", response_model=List[MetaLearningConfigResponse])
async def list_meta_learning_configurations(api_key: str = Depends(get_api_key)):
    """Get all meta-learning configurations for the current user"""
    return get_meta_learning_configurations(api_key)

# Optimization Strategy Endpoints
@router.get("/strategies")
async def get_optimization_strategies(api_key: str = Depends(get_api_key)):
    """Get available optimization strategies"""
    from app.optimizer import get_optimization_strategies
    return get_optimization_strategies()

# Visualization Data Endpoints
@router.get("/visualization/experiments")
async def get_experiment_visualization_data(api_key: str = Depends(get_api_key)):
    """Get experiment data for visualization"""
    # This would fetch data from the database to populate visualizations
    return {
        "experiments": [
            {
                "id": "exp123",
                "name": "Medical Diagnosis Optimization",
                "date": "2023-05-10T14:30:00Z",
                "iterations": 5,
                "metrics": {
                    "exact_match": [0.45, 0.52, 0.61, 0.68, 0.72],
                    "bleu": [0.38, 0.45, 0.53, 0.59, 0.64]
                }
            }
        ]
    }
