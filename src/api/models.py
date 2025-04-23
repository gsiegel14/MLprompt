
from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import uuid


class PromptCreate(BaseModel):
    system_prompt: str
    output_prompt: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    
class PromptResponse(BaseModel):
    prompt_id: str
    system_prompt: str
    output_prompt: str
    version: int
    parent_id: Optional[str] = None
    created_at: datetime
    metadata: Dict[str, Any]


class ModelProvider(str, Enum):
    VERTEX_AI = "vertex_ai"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


class InferenceRequest(BaseModel):
    system_prompt: Optional[str] = None
    output_prompt: Optional[str] = None
    prompt_id: Optional[str] = None
    user_input: str
    temperature: float = 0.0
    max_tokens: int = 1024
    model_provider: ModelProvider = ModelProvider.VERTEX_AI
    
    @validator('prompt_id', 'system_prompt', 'output_prompt')
    def validate_prompt_inputs(cls, v, values):
        if 'prompt_id' not in values or not values['prompt_id']:
            if not (('system_prompt' in values and values['system_prompt']) and 
                    ('output_prompt' in values and values['output_prompt'])):
                raise ValueError("Either prompt_id or both system_prompt and output_prompt must be provided")
        return v


class InferenceResponse(BaseModel):
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    response_text: str
    prompt_id: Optional[str] = None
    tokens_used: int
    generated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchInferenceRequest(BaseModel):
    prompt_id: Optional[str] = None
    system_prompt: Optional[str] = None
    output_prompt: Optional[str] = None
    inputs: List[str]
    temperature: float = 0.0
    max_tokens: int = 1024
    model_provider: ModelProvider = ModelProvider.VERTEX_AI


class BatchInferenceResponse(BaseModel):
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    responses: List[InferenceResponse]
    total_tokens: int
    generated_at: datetime = Field(default_factory=datetime.now)


class EvaluationMetric(str, Enum):
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    EXACT_MATCH = "exact_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CUSTOM = "custom"


class EvaluationRequest(BaseModel):
    predictions: List[str]
    ground_truths: List[str]
    metrics: List[EvaluationMetric] = [EvaluationMetric.ACCURACY]
    custom_evaluator: Optional[str] = None


class EvaluationResult(BaseModel):
    evaluation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metrics: Dict[str, float]
    details: Optional[Dict[str, Any]] = None
    evaluated_at: datetime = Field(default_factory=datetime.now)


class DatasetItem(BaseModel):
    input: str
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    items: List[DatasetItem]
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetResponse(BaseModel):
    dataset_id: str
    name: str
    description: Optional[str] = None
    item_count: int
    tags: List[str]
    created_at: datetime
    last_modified: datetime
    metadata: Dict[str, Any]


class DatasetSample(BaseModel):
    items: List[DatasetItem]
    dataset_id: str
    sample_size: int
    total_items: int


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationStrategy(str, Enum):
    FULL_REWRITE = "full_rewrite"
    TARGETED_EDIT = "targeted_edit"
    EXAMPLE_ADDITION = "example_addition"
    HYBRID = "hybrid"
    AUTO = "auto"


class ExperimentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    prompt_id: str
    dataset_id: str
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.AUTO
    max_iterations: int = 3
    evaluation_metrics: List[EvaluationMetric] = [EvaluationMetric.ACCURACY]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentResponse(BaseModel):
    experiment_id: str
    name: str
    description: Optional[str] = None
    prompt_id: str
    dataset_id: str
    status: ExperimentStatus
    current_iteration: int
    max_iterations: int
    created_at: datetime
    last_updated: datetime
    metadata: Dict[str, Any]


class ExperimentMetrics(BaseModel):
    experiment_id: str
    iterations: List[Dict[str, Any]]
    best_iteration: int
    best_metrics: Dict[str, float]
    comparison: Dict[str, Dict[str, float]]


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float


class CostSummary(BaseModel):
    time_period: str
    total_cost: float
    token_usage: TokenUsage
    breakdown_by_model: Dict[str, float]
    breakdown_by_endpoint: Dict[str, float]


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str
    expires_at: datetime


class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None
    scopes: List[str] = []
