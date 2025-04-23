
"""
API endpoints for prompt optimization
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
import logging
from typing import List, Dict, Any

from src.api.models import OptimizationRequest, OptimizationResult
from src.app.config import settings
from src.flows.prompt_optimization_flow import prompt_optimization_flow

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/optimize", response_model=OptimizationResult)
async def optimize_prompts(request: OptimizationRequest):
    """
    Run the 5-step prompt optimization flow
    
    1. Primary LLM Inference
    2. Hugging Face Evaluation 
    3. Optimizer LLM
    4. Refined LLM Inference
    5. Second Evaluation
    """
    try:
        # Convert examples to list of dicts
        examples = [example.dict() for example in request.examples]
        
        # Run the optimization flow
        result = prompt_optimization_flow(
            system_prompt=request.prompt_data.system_prompt,
            output_prompt=request.prompt_data.output_prompt,
            examples=examples,
            vertex_project_id=settings.VERTEX_PROJECT_ID,
            vertex_location=settings.VERTEX_LOCATION,
            primary_model_name=request.primary_model_name,
            optimizer_model_name=request.optimizer_model_name,
            metrics=request.metrics,
            target_threshold=request.target_threshold,
            max_iterations=request.max_iterations
        )
        
        return result
    except Exception as e:
        logger.error(f"Error in optimize_prompts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize-async", status_code=202)
async def optimize_prompts_async(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks
):
    """
    Run the 5-step prompt optimization flow asynchronously
    """
    try:
        # Generate a unique ID for this optimization run
        import uuid
        from datetime import datetime
        
        run_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Function to run in background
        def run_optimization():
            try:
                # Convert examples to list of dicts
                examples = [example.dict() for example in request.examples]
                
                # Run the optimization flow
                result = prompt_optimization_flow(
                    system_prompt=request.prompt_data.system_prompt,
                    output_prompt=request.prompt_data.output_prompt,
                    examples=examples,
                    vertex_project_id=settings.VERTEX_PROJECT_ID,
                    vertex_location=settings.VERTEX_LOCATION,
                    primary_model_name=request.primary_model_name,
                    optimizer_model_name=request.optimizer_model_name,
                    metrics=request.metrics,
                    target_threshold=request.target_threshold,
                    max_iterations=request.max_iterations
                )
                
                # Save results to a file
                import json
                import os
                
                # Create experiments directory if it doesn't exist
                os.makedirs(f"experiments/{timestamp}", exist_ok=True)
                
                # Save results
                with open(f"experiments/{timestamp}/result.json", "w") as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"Optimization run {run_id} completed successfully")
            except Exception as e:
                logger.error(f"Error in background optimization task: {str(e)}")
        
        # Add task to background tasks
        background_tasks.add_task(run_optimization)
        
        return {
            "run_id": run_id,
            "timestamp": timestamp,
            "status": "optimization_started",
            "message": "Optimization started in the background"
        }
    except Exception as e:
        logger.error(f"Error in optimize_prompts_async: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends

from src.app.clients.vertex_client import VertexClient
from src.app.utils.hyperparameter_tuner import HyperparameterTuner
from app.cross_validator import CrossValidator
from app.data_module import DataModule
from app.workflow import PromptOptimizationWorkflow
from app.experiment_tracker import ExperimentTracker
from app.optimizer import get_optimization_strategies

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize components
data_module = DataModule()
experiment_tracker = ExperimentTracker()
workflow = PromptOptimizationWorkflow(data_module, experiment_tracker)
hyperparameter_tuner = HyperparameterTuner()
cross_validator = CrossValidator(data_module, workflow)

@router.post("/hyperparameter-tune")
async def run_hyperparameter_tuning(
    system_prompt: str,
    output_prompt: str,
    search_space: Optional[Dict[str, List]] = None,
    metric_key: str = 'best_score'
):
    """
    Run hyperparameter tuning to find optimal configuration.
    
    Args:
        system_prompt: System prompt to use
        output_prompt: Output prompt to use
        search_space: Optional custom search space
        metric_key: Metric to optimize
        
    Returns:
        Dict with tuning results
    """
    try:
        logger.info("Starting hyperparameter tuning")
        
        results = hyperparameter_tuner.run_grid_search(
            workflow=workflow,
            system_prompt=system_prompt,
            output_prompt=output_prompt,
            search_space=search_space,
            metric_key=metric_key
        )
        
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {e}")
        raise HTTPException(status_code=500, detail=f"Error in hyperparameter tuning: {str(e)}")

@router.post("/cross-validate")
async def run_cross_validation(
    system_prompt: str,
    output_prompt: str,
    fold_count: int = 5
):
    """
    Run cross-validation for a prompt system.
    
    Args:
        system_prompt: System prompt to validate
        output_prompt: Output prompt to validate
        fold_count: Number of folds for cross-validation
        
    Returns:
        Dict with validation results
    """
    try:
        logger.info(f"Starting {fold_count}-fold cross-validation")
        
        # Update fold count
        cross_validator.folds = fold_count
        
        # Get all available examples
        train_examples = data_module.get_train_examples() or []
        validation_examples = data_module.get_validation_examples() or []
        all_examples = train_examples + validation_examples
        
        if not all_examples:
            return {
                "status": "error",
                "message": "No examples available for cross-validation"
            }
        
        results = cross_validator.evaluate_system(
            system_prompt=system_prompt,
            output_prompt=output_prompt,
            examples=all_examples
        )
        
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in cross-validation: {e}")
        raise HTTPException(status_code=500, detail=f"Error in cross-validation: {str(e)}")

@router.post("/compare-systems")
async def compare_prompt_systems(
    systems: List[Dict[str, str]],
    fold_count: int = 5
):
    """
    Compare multiple prompt systems using cross-validation.
    
    Args:
        systems: List of dicts with 'system_prompt', 'output_prompt', and 'name'
        fold_count: Number of folds for cross-validation
        
    Returns:
        Dict with comparison results
    """
    try:
        logger.info(f"Starting comparison of {len(systems)} prompt systems")
        
        # Update fold count
        cross_validator.folds = fold_count
        
        results = cross_validator.compare_systems(systems)
        
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in system comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Error in system comparison: {str(e)}")

@router.get("/optimization-strategies")
async def get_available_strategies():
    """Get available optimization strategies."""
    try:
        strategies = get_optimization_strategies()
        return {"strategies": strategies}
    except Exception as e:
        logger.error(f"Error getting optimization strategies: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting optimization strategies: {str(e)}")
