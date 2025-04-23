
from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security import SecurityScopes
from typing import List, Optional

from src.api.models import (
    InferenceRequest, InferenceResponse, 
    BatchInferenceRequest, BatchInferenceResponse,
    EvaluationRequest, EvaluationResult
)
from src.app.clients.vertex_client import VertexAIClient
from src.app.clients.hf_evaluator import HuggingFaceEvaluator
from src.app.auth import get_current_active_user
from src.app.utils.cost_tracker import track_token_usage

router = APIRouter(prefix="/inference", tags=["Inference"])

@router.post("/complete", response_model=InferenceResponse)
async def generate_completion(
    request: InferenceRequest,
    current_user = Security(get_current_active_user, scopes=["inference"])
):
    """
    Generate a completion for a single prompt input
    """
    try:
        # Initialize client based on selected provider
        if request.model_provider == "vertex_ai":
            client = VertexAIClient()
        else:
            # For simplicity, defaulting to Vertex AI
            # In a real implementation, would have different client implementations
            client = VertexAIClient()
            
        # Get system and output prompts
        system_prompt = request.system_prompt
        output_prompt = request.output_prompt
        
        # If prompt_id provided, fetch prompts from database
        if not system_prompt or not output_prompt:
            if request.prompt_id:
                # This would fetch from database in real implementation
                # Mocked for now
                system_prompt = "You are an AI assistant that helps with various tasks."
                output_prompt = "Provide a helpful response to: {{user_input}}"
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="Either prompt_id or both system_prompt and output_prompt must be provided"
                )
        
        # Format the output prompt with user input
        formatted_output_prompt = output_prompt.replace("{{user_input}}", request.user_input)
        
        # Generate response
        response_text, token_info = await client.generate_text(
            system_prompt=system_prompt,
            prompt=formatted_output_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Track token usage
        track_token_usage(
            tokens=token_info.total_tokens,
            model=client.model_name,
            endpoint="inference_complete",
            user_id=current_user.user_id
        )
        
        # Create response
        response = InferenceResponse(
            response_text=response_text,
            prompt_id=request.prompt_id,
            tokens_used=token_info.total_tokens,
            metadata={
                "model": client.model_name,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "prompt_tokens": token_info.prompt_tokens,
                "completion_tokens": token_info.completion_tokens
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchInferenceResponse)
async def batch_completion(
    request: BatchInferenceRequest,
    current_user = Security(get_current_active_user, scopes=["inference"])
):
    """
    Generate completions for a batch of inputs using the same prompt
    """
    try:
        # Initialize client
        client = VertexAIClient()
            
        # Get system and output prompts
        system_prompt = request.system_prompt
        output_prompt = request.output_prompt
        
        # If prompt_id provided, fetch prompts from database
        if not system_prompt or not output_prompt:
            if request.prompt_id:
                # This would fetch from database in real implementation
                # Mocked for now
                system_prompt = "You are an AI assistant that helps with various tasks."
                output_prompt = "Provide a helpful response to: {{user_input}}"
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="Either prompt_id or both system_prompt and output_prompt must be provided"
                )
        
        # Process each input
        responses = []
        total_tokens = 0
        
        for user_input in request.inputs:
            # Format the output prompt with user input
            formatted_output_prompt = output_prompt.replace("{{user_input}}", user_input)
            
            # Generate response
            response_text, token_info = await client.generate_text(
                system_prompt=system_prompt,
                prompt=formatted_output_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Add to responses
            response = InferenceResponse(
                response_text=response_text,
                prompt_id=request.prompt_id,
                tokens_used=token_info.total_tokens,
                metadata={
                    "input": user_input,
                    "model": client.model_name,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens
                }
            )
            
            responses.append(response)
            total_tokens += token_info.total_tokens
        
        # Track token usage
        track_token_usage(
            tokens=total_tokens,
            model=client.model_name,
            endpoint="inference_batch",
            user_id=current_user.user_id
        )
        
        # Create batch response
        batch_response = BatchInferenceResponse(
            responses=responses,
            total_tokens=total_tokens
        )
        
        return batch_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate", response_model=EvaluationResult)
async def evaluate_predictions(
    request: EvaluationRequest,
    current_user = Security(get_current_active_user, scopes=["inference"])
):
    """
    Evaluate model predictions against ground truth
    """
    try:
        # Initialize evaluator
        evaluator = HuggingFaceEvaluator()
        
        # Validate inputs
        if len(request.predictions) != len(request.ground_truths):
            raise HTTPException(
                status_code=400, 
                detail="Number of predictions must match number of ground truths"
            )
        
        # Run evaluation
        metrics = {}
        details = {}
        
        for metric in request.metrics:
            if metric == "semantic_similarity":
                score, metric_details = await evaluator.evaluate_semantic_similarity(
                    predictions=request.predictions,
                    references=request.ground_truths
                )
                metrics[metric] = score
                details[metric] = metric_details
            elif metric == "exact_match":
                score = await evaluator.evaluate_exact_match(
                    predictions=request.predictions,
                    references=request.ground_truths
                )
                metrics[metric] = score
            elif metric == "custom" and request.custom_evaluator:
                # Custom evaluation using provided evaluator prompt
                # This would be implemented with a custom evaluation function
                score = 0.5  # Placeholder
                metrics[metric] = score
            else:
                # Default to exact match for other metrics
                score = await evaluator.evaluate_exact_match(
                    predictions=request.predictions,
                    references=request.ground_truths
                )
                metrics[metric] = score
        
        # Create response
        result = EvaluationResult(
            metrics=metrics,
            details=details
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
