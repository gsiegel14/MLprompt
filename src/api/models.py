"""
API models and schemas for the Prompt Optimization Platform
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

# ML Settings Models
class ModelConfigurationCreate(BaseModel):
    name: str
    primary_model: str
    optimizer_model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    top_k: int = 40
    is_default: bool = False

class ModelConfigurationUpdate(BaseModel):
    name: Optional[str] = None
    primary_model: Optional[str] = None
    optimizer_model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    is_default: Optional[bool] = None

class ModelConfigurationResponse(BaseModel):
    id: str
    name: str
    primary_model: str
    optimizer_model: str
    temperature: float
    max_tokens: int
    top_p: float
    top_k: int
    is_default: bool

class MetricConfigurationCreate(BaseModel):
    name: str
    metrics: List[str]
    metric_weights: Dict[str, float] = Field(default_factory=dict)
    target_threshold: float = 0.8

class MetricConfigurationUpdate(BaseModel):
    name: Optional[str] = None
    metrics: Optional[List[str]] = None
    metric_weights: Optional[Dict[str, float]] = None
    target_threshold: Optional[float] = None

class MetricConfigurationResponse(BaseModel):
    id: str
    name: str
    metrics: List[str]
    metric_weights: Dict[str, float]
    target_threshold: float

class MetaLearningConfigurationCreate(BaseModel):
    name: str
    model_type: str = "xgboost"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    feature_selection: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True

class MetaLearningConfigurationUpdate(BaseModel):
    name: Optional[str] = None
    model_type: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    feature_selection: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class MetaLearningConfigurationResponse(BaseModel):
    id: str
    name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    feature_selection: Dict[str, Any]
    is_active: bool

# Base models
class IDModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# User models
class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: str
    is_active: bool

    class Config:
        orm_mode = True

# Authentication models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Prompt models
class PromptBase(BaseModel):
    system_prompt: str
    output_prompt: str

class PromptCreate(PromptBase):
    name: str
    description: Optional[str] = None

class Prompt(PromptBase):
    id: str
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    user_id: str

    class Config:
        orm_mode = True

# Dataset models
class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None

class DatasetCreate(DatasetBase):
    examples: List[Dict[str, str]]

class Dataset(DatasetBase):
    id: str
    created_at: datetime
    updated_at: datetime
    user_id: str
    example_count: int

    class Config:
        orm_mode = True

# Experiment models
class ExperimentBase(BaseModel):
    name: str
    description: Optional[str] = None

class ExperimentCreate(ExperimentBase):
    prompt_id: str
    dataset_id: str
    iterations: int = 3

class Experiment(ExperimentBase):
    id: str
    prompt_id: str
    dataset_id: str
    iterations: int
    status: str
    created_at: datetime
    updated_at: datetime
    user_id: str

    class Config:
        orm_mode = True

# Optimization request and result models
class OptimizationRequest(BaseModel):
    system_prompt: str
    output_prompt: str
    examples: List[Dict[str, str]]
    iterations: int = 3
    metrics: List[str] = ["exact_match"]

class OptimizationIteration(BaseModel):
    iteration: int
    system_prompt: str
    output_prompt: str
    metrics: Dict[str, float]
    reasoning: Optional[str] = None

class OptimizationResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_system_prompt: str
    original_output_prompt: str
    iterations: List[OptimizationIteration]
    final_system_prompt: str
    final_output_prompt: str
    final_metrics: Dict[str, float]
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Inference models
class InferenceRequest(BaseModel):
    system_prompt: str
    output_prompt: str
    inputs: Dict[str, str]

class InferenceResponse(BaseModel):
    output: str
    metrics: Optional[Dict[str, float]] = None

# ML Settings models

class ModelConfigurationBase(BaseModel):
    name: str
    primary_model: str
    optimizer_model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    top_k: int = 40
    is_default: bool = False

class ModelConfigurationCreate(ModelConfigurationBase):
    pass

class ModelConfigurationResponse(ModelConfigurationBase):
    id: str
    user_id: str

    class Config:
        orm_mode = True

class MetricConfigurationBase(BaseModel):
    name: str
    metrics: List[str]
    metric_weights: Dict[str, float] = Field(default_factory=dict)
    target_threshold: float = 0.8

class MetricConfigurationCreate(MetricConfigurationBase):
    pass

class MetricConfigurationResponse(MetricConfigurationBase):
    id: str
    user_id: str

    class Config:
        orm_mode = True

class MetaLearningConfigurationBase(BaseModel):
    name: str
    model_type: str = "xgboost"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    feature_selection: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True

class MetaLearningConfigurationCreate(MetaLearningConfigurationBase):
    pass

class MetaLearningConfigurationResponse(MetaLearningConfigurationBase):
    id: str
    user_id: str
    last_trained: Optional[datetime] = None
    performance: Optional[float] = None
    model_path: Optional[str] = None

    class Config:
        orm_mode = True

# Experiment visualization models
class PromptVersion(BaseModel):
    version: int
    system_prompt: str
    output_prompt: str

class ExperimentMetrics(BaseModel):
    metrics_history: List[Dict[str, float]]
    prompt_versions: List[PromptVersion]