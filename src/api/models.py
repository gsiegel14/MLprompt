
"""
Pydantic models for API request and response validation
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PromptData(BaseModel):
    """Request model for prompt creation/update"""
    system_prompt: str
    output_prompt: str
    name: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

class PromptResponse(BaseModel):
    """Response model for prompt data"""
    id: str
    system_prompt: str
    output_prompt: str
    version: int
    name: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: str
    updated_at: str

class OptimizationRequest(BaseModel):
    """Request model for starting an optimization workflow"""
    prompt_id: str
    dataset_id: str
    target_metric: str = "exact_match_score"
    target_threshold: float = 0.9
    max_iterations: int = 10
    patience: int = 3
    primary_model: Optional[str] = None
    optimizer_model: Optional[str] = None

class OptimizationResponse(BaseModel):
    """Response model for optimization status"""
    flow_id: str
    status: str
    prompt_id: str
    dataset_id: str
    started_at: str
    completed_at: Optional[str] = None
    current_iteration: int = 0
    max_iterations: int
    best_score: Optional[float] = None

class EvaluationRequest(BaseModel):
    """Request model for prompt evaluation"""
    prompt_id: str
    dataset_id: str
    metrics: List[str] = Field(default_factory=lambda: ["exact_match", "semantic_similarity"])

class EvaluationResponse(BaseModel):
    """Response model for evaluation results"""
    prompt_id: str
    dataset_id: str
    metrics: Dict[str, float]
    examples_count: int
    timestamp: str

class DatasetUploadRequest(BaseModel):
    """Request model for dataset upload"""
    name: str
    description: Optional[str] = None
    train_split: float = 0.8

class DatasetResponse(BaseModel):
    """Response model for dataset info"""
    id: str
    name: str
    description: Optional[str] = None
    examples_count: int
    train_count: int
    validation_count: int
    created_at: str
    updated_at: str
