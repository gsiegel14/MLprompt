"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class PromptData(BaseModel):
    """Request model for creating or updating prompts"""
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
    name: str
    description: str
    tags: List[str]
    created_at: str
    updated_at: str

class OptimizationRequest(BaseModel):
    """Request model for starting a prompt optimization workflow"""
    prompt_id: str
    dataset_id: str
    primary_model: Optional[str] = None
    optimizer_model: Optional[str] = None
    target_metric: str = "exact_match_score"
    target_threshold: float = 0.9
    patience: int = 3
    max_iterations: int = 10

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
    target_metric: str
    target_threshold: float
    patience: int
    primary_model: str
    optimizer_model: str

class InferenceRequest(BaseModel):
    """Request model for running inference"""
    prompt_id: str
    user_input: str
    model_name: Optional[str] = None
    temperature: float = 0.7

class InferenceResponse(BaseModel):
    """Response model for inference results"""
    prompt_id: str
    user_input: str
    generated_text: str
    model_name: str
    timestamp: str

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