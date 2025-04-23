
"""
API endpoints for inference and evaluation
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
import logging
from typing import List, Dict, Any, Optional

from src.api.models import InferenceRequest, InferenceResult, Example
from src.app.auth import get_current_active_user, User
from src.app.clients.vertex_client import VertexAIClient
from src.app.clients.hf_evaluator import HuggingFaceEvaluator
from src.app.config import settings

router = APIRouter(prefix="/inference", tags=["inference"])
logger = logging.getLogger(__name__)

@router.post("/complete", response_model=InferenceResult)
async def run_inference(
    request: InferenceRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Run inference on examples using the specified prompt and model"""
    try:
        # Initialize Vertex AI client
        vertex_client = VertexAIClient(
            project_id=settings.VERTEX_PROJECT_ID,
            location=settings.VERTEX_LOCATION
        )
        
        # Prepare results
        results = []
        
        # Process each example
        for example in request.examples:
            # Skip if no user input
            if not example.user_input:
                continue
            
            # Run inference
            response = await vertex_client.generate_text(
                system_prompt=request.prompt_data.system_prompt,
                output_prompt=request.prompt_data.output_prompt,
                user_input=example.user_input,
                model_name=request.model_name
            )
            
            # Add to results
            results.append({
                "user_input": example.user_input,
                "ground_truth_output": example.ground_truth_output,
                "model_output": response
            })
        
        # Return results
        return {
            "examples": results
        }
    except Exception as e:
        logger.error(f"Error in run_inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=InferenceResult)
async def run_batch_inference(
    request: InferenceRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Run batch inference (async) on examples using the specified prompt and model"""
    try:
        # Generate a unique ID for this batch
        import uuid
        from datetime import datetime
        
        batch_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Function to run in background
        async def run_inference_batch():
            try:
                # Initialize Vertex AI client
                vertex_client = VertexAIClient(
                    project_id=settings.VERTEX_PROJECT_ID,
                    location=settings.VERTEX_LOCATION
                )
                
                # Prepare results
                results = []
                
                # Process each example
                for example in request.examples:
                    # Skip if no user input
                    if not example.user_input:
                        continue
                    
                    # Run inference
                    response = await vertex_client.generate_text(
                        system_prompt=request.prompt_data.system_prompt,
                        output_prompt=request.prompt_data.output_prompt,
                        user_input=example.user_input,
                        model_name=request.model_name
                    )
                    
                    # Add to results
                    results.append({
                        "user_input": example.user_input,
                        "ground_truth_output": example.ground_truth_output,
                        "model_output": response
                    })
                
                # Save results to file
                import json
                import os
                
                # Create batch directory if it doesn't exist
                os.makedirs(f"batch_results/{timestamp}", exist_ok=True)
                
                # Save results
                with open(f"batch_results/{timestamp}/results.json", "w") as f:
                    json.dump({"examples": results}, f, indent=2)
                
                logger.info(f"Batch inference {batch_id} completed successfully")
            except Exception as e:
                logger.error(f"Error in background inference task: {str(e)}")
        
        # Add task to background tasks
        background_tasks.add_task(run_inference_batch)
        
        return {
            "batch_id": batch_id,
            "timestamp": timestamp,
            "status": "inference_started",
            "message": "Batch inference started in the background"
        }
    except Exception as e:
        logger.error(f"Error in run_batch_inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate", response_model=Dict[str, Any])
async def evaluate_results(
    examples: List[Example],
    metrics: List[str] = ["exact_match", "semantic_similarity"],
    current_user: User = Depends(get_current_active_user)
):
    """Evaluate model outputs against ground truth"""
    try:
        # Initialize HuggingFace evaluator
        hf_evaluator = HuggingFaceEvaluator()
        
        # Prepare data for evaluation
        predictions = []
        references = []
        
        for example in examples:
            if example.model_output and example.ground_truth_output:
                predictions.append(example.model_output)
                references.append(example.ground_truth_output)
        
        if not predictions:
            raise HTTPException(
                status_code=400, 
                detail="No valid examples found for evaluation (need both model_output and ground_truth_output)"
            )
        
        # Evaluate
        results = {}
        for metric_name in metrics:
            metric_result = hf_evaluator.evaluate(
                predictions=predictions,
                references=references,
                metric_name=metric_name
            )
            results[metric_name] = metric_result
        
        # Calculate average scores
        avg_scores = {}
        for metric_name, metric_results in results.items():
            if isinstance(metric_results, list):
                avg_scores[metric_name] = sum(metric_results) / len(metric_results)
            elif isinstance(metric_results, dict) and "scores" in metric_results:
                avg_scores[metric_name] = sum(metric_results["scores"]) / len(metric_results["scores"])
        
        return {
            "metrics": results,
            "average_scores": avg_scores,
            "example_count": len(predictions)
        }
    except Exception as e:
        logger.error(f"Error in evaluate_results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
