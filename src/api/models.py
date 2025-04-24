from pydantic import BaseModel, Field, UUID4
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid

# Base model classes
class ModelBase(BaseModel):
    class Config:
        from_attributes = True

# Prompt models
class PromptBase(ModelBase):
    system_prompt: str
    output_prompt: str

class PromptCreate(PromptBase):
    parent_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PromptResponse(PromptBase):
    id: UUID4
    parent_id: Optional[UUID4] = None
    version: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

# Dataset models
class DatasetBase(ModelBase):
    name: str
    file_path: str

class DatasetCreate(DatasetBase):
    row_count: Optional[int] = None
    columns: Optional[Dict[str, str]] = None

class DatasetResponse(DatasetBase):
    id: UUID4
    row_count: Optional[int] = None
    columns: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime

# Experiment models
class ExperimentBase(ModelBase):
    name: str
    initial_prompt_id: str
    dataset_id: str

class ExperimentCreate(ExperimentBase):
    metrics: List[str] = Field(default_factory=list)
    max_epochs: int = 10
    target_threshold: float = 0.8

class ExperimentResponse(ExperimentBase):
    id: UUID4
    metrics: List[str] = Field(default_factory=list)
    max_epochs: int
    target_threshold: float
    status: str
    best_prompt_id: Optional[UUID4] = None
    created_at: datetime

class ExperimentMetricsResponse(ModelBase):
    experiment_id: UUID4
    epoch: int
    metrics: Dict[str, Any]
    prompt_id: Optional[UUID4] = None
    created_at: datetime

# ML Settings models
class ModelConfigBase(ModelBase):
    name: str
    primary_model: str
    optimizer_model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    top_k: int = 40

class ModelConfigCreate(ModelConfigBase):
    is_default: bool = False

class ModelConfigResponse(ModelConfigBase):
    id: UUID4
    is_default: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

class ModelConfigUpdate(ModelBase):
    name: Optional[str] = None
    primary_model: Optional[str] = None
    optimizer_model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    is_default: Optional[bool] = None

# Optimization models
class PromptOptimizationRequest(ModelBase):
    system_prompt_path: str
    output_prompt_path: str
    dataset_path: str
    metric_names: List[str] = ["exact_match", "bleu"]
    target_metric: str = "avg_score"
    target_threshold: float = 0.8
    patience: int = 3
    max_iterations: int = 5
    batch_size: int = 5
    sample_k: int = 3
    optimizer_strategy: str = "reasoning_first"

class PromptOptimizationResponse(ModelBase):
    experiment_id: str
    status: str
    message: str