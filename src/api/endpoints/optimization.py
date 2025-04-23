
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
