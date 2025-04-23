"""
API data models
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import uuid


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OptimizationStrategy(str, Enum):
    REASONING_FIRST = "reasoning_first"
    FULL_REWRITE = "full_rewrite"
    TARGETED_EDIT = "targeted_edit"
    EXAMPLE_ADDITION = "example_addition"
    MEDICAL_DIAGNOSTIC = "medical_diagnostic"


class PromptBase(BaseModel):
    """Base model for prompts"""
    system_prompt: str
    output_prompt: str
    version: int = 1


class PromptCreate(PromptBase):
    """Model for creating new prompts"""
    pass


class PromptResponse(PromptBase):
    """Response model for returning prompts"""
    id: str
    created_at: datetime
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

    class Config:
        from_attributes = True


class ExampleBase(BaseModel):
    """Base model for examples"""
    user_input: str
    ground_truth_output: str


class ExampleCreate(ExampleBase):
    """Model for creating new examples"""
    pass


class ExampleResponse(ExampleBase):
    """Response model for returning examples"""
    id: str
    model_response: Optional[str] = None
    score: Optional[float] = None

    class Config:
        from_attributes = True


class DatasetBase(BaseModel):
    """Base model for datasets"""
    name: str
    description: Optional[str] = None


class DatasetCreate(DatasetBase):
    """Model for creating new datasets"""
    examples: List[ExampleCreate]


class DatasetResponse(DatasetBase):
    """Response model for returning datasets"""
    id: str
    created_at: datetime
    example_count: int

    class Config:
        from_attributes = True


class OptimizationJobBase(BaseModel):
    """Base model for optimization jobs"""
    system_prompt_id: str
    output_prompt_id: str
    dataset_id: str
    metric_names: List[str] = ["exact_match", "bleu"]
    target_metric: str = "exact_match"
    target_threshold: float = 0.9
    max_iterations: int = 3
    strategy: OptimizationStrategy = OptimizationStrategy.REASONING_FIRST


class OptimizationJobCreate(OptimizationJobBase):
    """Model for creating new optimization jobs"""
    pass


class OptimizationJobResponse(OptimizationJobBase):
    """Response model for returning optimization jobs"""
    id: str
    status: ExperimentStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    best_prompt_id: Optional[str] = None
    iterations_completed: int = 0
    best_metrics: Dict[str, float] = {}

    class Config:
        from_attributes = True


class ModelConfigBase(BaseModel):
    """Base model for ML model configurations"""
    name: str
    primary_model: str
    optimizer_model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    top_k: int = 40
    is_default: bool = False


class ModelConfigCreate(ModelConfigBase):
    """Model for creating new ML model configurations"""
    pass


class ModelConfigResponse(ModelConfigBase):
    """Response model for returning ML model configurations"""
    id: str
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True


class MetricConfigBase(BaseModel):
    """Base model for metric configurations"""
    name: str
    metrics: List[str]
    metric_weights: Dict[str, float] = {}
    target_threshold: float = 0.8


class MetricConfigCreate(MetricConfigBase):
    """Model for creating new metric configurations"""
    pass


class MetricConfigResponse(MetricConfigBase):
    """Response model for returning metric configurations"""
    id: str
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True


class MetaLearningConfigBase(BaseModel):
    """Base model for meta-learning configurations"""
    name: str
    model_type: str = "xgboost"
    hyperparameters: Dict[str, Any] = {}
    feature_selection: Dict[str, Any] = {}
    is_active: bool = True


class MetaLearningConfigCreate(MetaLearningConfigBase):
    """Model for creating new meta-learning configurations"""
    pass


class MetaLearningConfigResponse(MetaLearningConfigBase):
    """Response model for returning meta-learning configurations"""
    id: str
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True


class ExperimentBase(BaseModel):
    """Base model for experiments"""
    name: str
    description: Optional[str] = None
    model_config_id: str
    metric_config_id: str
    dataset_id: str
    max_iterations: int = 5
    strategy: OptimizationStrategy = OptimizationStrategy.REASONING_FIRST


class ExperimentCreate(ExperimentBase):
    """Model for creating new experiments"""
    pass


class ExperimentResponse(ExperimentBase):
    """Response model for returning experiments"""
    id: str
    user_id: Optional[str] = None
    status: ExperimentStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    iterations_completed: int = 0
    current_best_score: float = 0.0
    improvement_percentage: float = 0.0

    class Config:
        from_attributes = True