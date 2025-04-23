"""
Pydantic models for the API endpoints
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Example(BaseModel):
    """A single example for training or evaluation"""
    user_input: str
    ground_truth_output: Optional[str] = None
    model_output: Optional[str] = None
    optimized_output: Optional[str] = None

class PromptData(BaseModel):
    """Prompt information for submission"""
    system_prompt: str
    output_prompt: str
    version: int = 1

class OptimizationRequest(BaseModel):
    """Request to run prompt optimization"""
    prompt_data: PromptData
    examples: List[Example]
    primary_model_name: Optional[str] = Field(default="gemini-1.5-flash-001")
    optimizer_model_name: Optional[str] = Field(default="gemini-1.5-pro-001")
    metrics: Optional[List[str]] = Field(default=["exact_match", "semantic_similarity"])
    max_iterations: Optional[int] = Field(default=3)
    target_threshold: Optional[float] = Field(default=0.9)

class OptimizationResult(BaseModel):
    """Results from a prompt optimization run"""
    best_prompt_state: Dict[str, Any]
    best_metrics: Dict[str, Any]
    history: List[Dict[str, Any]]
    iterations_completed: int

class InferenceRequest(BaseModel):
    """Request to run inference with a prompt on examples"""
    prompt_data: PromptData
    examples: List[Example]
    model_name: Optional[str] = Field(default="gemini-1.5-flash-001")

class InferenceResult(BaseModel):
    """Results from an inference run"""
    examples: List[Example]
    metrics: Optional[Dict[str, Any]] = None

class ExperimentData(BaseModel):
    """Information about a prompt optimization experiment"""
    id: str
    created_at: str
    iterations: int
    best_metrics: Dict[str, Any]
    best_prompt_state: Dict[str, Any]

class ExperimentList(BaseModel):
    """List of experiments"""
    experiments: List[ExperimentData]