"""
Prompt Optimization Flow using Prefect 2.0
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

from src.app.models.prompt_state import PromptState
from src.app.clients.vertex_client import VertexAIClient
from src.app.clients.hf_evaluator import EvaluatorService

logger = logging.getLogger(__name__)

@task(name="load_prompt_state")
def load_prompt_state(system_prompt: str, output_prompt: str, version: int = 1) -> PromptState:
    """
    Load or create a prompt state

    Args:
        system_prompt: System prompt text or path to file
        output_prompt: Output prompt text or path to file
        version: Version number

    Returns:
        PromptState object
    """
    # Check if inputs are file paths
    if system_prompt.endswith('.txt') and output_prompt.endswith('.txt'):
        try:
            with open(system_prompt, 'r') as f:
                system_prompt_text = f.read()
            with open(output_prompt, 'r') as f:
                output_prompt_text = f.read()

            return PromptState(
                system_prompt=system_prompt_text,
                output_prompt=output_prompt_text,
                version=version,
                created_at=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Error loading prompt files: {str(e)}")
            raise

    # Otherwise treat as direct text
    return PromptState(
        system_prompt=system_prompt,
        output_prompt=output_prompt,
        version=version,
        created_at=datetime.now().isoformat()
    )

@task(name="primary_llm_inference", retries=2)
def primary_llm_inference(
    prompt_state: Dict[str, Any],
    examples: List[Dict[str, str]],
    vertex_project_id: str,
    vertex_location: str,
    model_name: str
) -> List[Dict[str, Any]]:
    """
    Step 1: Generate baseline responses with current prompts

    Args:
        prompt_state: Dictionary representation of PromptState
        examples: List of examples to run inference on
        vertex_project_id: Google Cloud project ID
        vertex_location: Google Cloud region
        model_name: Name of primary LLM model to use

    Returns:
        List of examples with predictions
    """
    try:
        # Initialize Vertex client
        client = VertexAIClient(vertex_project_id, vertex_location)

        # Recreate PromptState from dict
        state = PromptState(**prompt_state)

        # Run batch prediction
        prediction_results = client.batch_predict(
            examples=examples,
            prompt_state={
                "system_prompt": state.system_prompt,
                "output_prompt": state.output_prompt
            },
            model_name=model_name
        )

        # Add predictions to examples
        for i, example in enumerate(examples):
            if i < len(prediction_results["predictions"]):
                example["model_output"] = prediction_results["predictions"][i]

        return examples
    except Exception as e:
        logger.error(f"Error in primary inference: {str(e)}")
        raise

@task(name="evaluate_baseline", retries=1)
def evaluate_baseline(
    examples: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Step 2: Compute baseline metrics using Hugging Face Evaluate

    Args:
        examples: List of examples with model predictions
        metrics: List of metrics to compute

    Returns:
        Dictionary of metric scores
    """
    try:
        # Extract predictions and references
        predictions = [ex.get("model_output", "") for ex in examples]
        references = [ex.get("ground_truth_output", "") for ex in examples]

        # Initialize evaluator
        evaluator = EvaluatorService()

        # Compute metrics
        results = evaluator.evaluate(
            predictions=predictions,
            references=references,
            metrics=metrics
        )

        # Add number of examples
        results["total_examples"] = len(examples)

        return results
    except Exception as e:
        logger.error(f"Error in baseline evaluation: {str(e)}")
        raise

@task(name="optimizer_llm", retries=2)
def optimizer_llm(
    prompt_state: Dict[str, Any],
    baseline_metrics: Dict[str, float],
    example_sample: List[Dict[str, Any]],
    vertex_project_id: str,
    vertex_location: str,
    optimizer_model_name: str
) -> Dict[str, Any]:
    """
    Step 3: Generate refined prompts based on performance data

    Args:
        prompt_state: Dictionary representation of PromptState
        baseline_metrics: Metrics from baseline evaluation
        example_sample: Sample of examples with predictions for analysis
        vertex_project_id: Google Cloud project ID
        vertex_location: Google Cloud region
        optimizer_model_name: Name of optimizer LLM model

    Returns:
        Updated PromptState with refined prompts
    """
    try:
        # Initialize Vertex client
        client = VertexAIClient(vertex_project_id, vertex_location)

        # Create optimizer prompt
        optimizer_prompt = f"""
As an AI prompt optimization expert, analyze the current prompts and performance metrics to create improved prompts.

CURRENT SYSTEM PROMPT:
{prompt_state["system_prompt"]}

CURRENT OUTPUT PROMPT:
{prompt_state["output_prompt"]}

PERFORMANCE METRICS:
{baseline_metrics}

EXAMPLE INPUTS AND OUTPUTS:
{example_sample[:5]}  # Include only a few examples to avoid overwhelming the model

Based on the above, please generate improved versions of both prompts. Focus on:
1. Fixing any issues visible in the example outputs
2. Enhancing clarity and specificity
3. Improving factual accuracy and reasoning

Respond with a JSON object containing:
{{"system_prompt": "improved system prompt here", "output_prompt": "improved output prompt here", "reasoning": "explanation of your changes here"}}
"""

        # Generate improved prompts
        response = client.generate_response(
            model_name=optimizer_model_name,
            user_content=optimizer_prompt,
            temperature=0.7
        )

        # Parse response (assuming it's valid JSON)
        try:
            import json
            optimized = json.loads(response)

            # Create new PromptState
            new_state = PromptState(
                system_prompt=optimized.get("system_prompt", prompt_state["system_prompt"]),
                output_prompt=optimized.get("output_prompt", prompt_state["output_prompt"]),
                version=prompt_state["version"] + 1,
                parent_id=prompt_state.get("id"),
                created_at=datetime.now().isoformat()
            )

            # Add reasoning to metadata
            new_state.metadata["optimization_reasoning"] = optimized.get("reasoning", "No reasoning provided")

            return new_state.dict()
        except json.JSONDecodeError:
            logger.error(f"Failed to parse optimizer response as JSON: {response[:100]}...")
            # Fall back to returning the original state with a version bump
            state = PromptState(**prompt_state)
            state.version += 1
            return state.dict()

    except Exception as e:
        logger.error(f"Error in optimizer LLM: {str(e)}")
        raise

@task(name="refined_llm_inference", retries=2)
def refined_llm_inference(
    optimized_prompt_state: Dict[str, Any],
    examples: List[Dict[str, Any]],
    vertex_project_id: str,
    vertex_location: str,
    model_name: str
) -> List[Dict[str, Any]]:
    """
    Step 4: Run inference with optimized prompts

    Args:
        optimized_prompt_state: Dictionary representation of optimized PromptState
        examples: List of examples to run inference on
        vertex_project_id: Google Cloud project ID
        vertex_location: Google Cloud region
        model_name: Name of primary LLM model to use

    Returns:
        List of examples with optimized predictions
    """
    try:
        # Initialize Vertex client
        client = VertexAIClient(vertex_project_id, vertex_location)

        # Run batch prediction with optimized prompts
        prediction_results = client.batch_predict(
            examples=examples,
            prompt_state={
                "system_prompt": optimized_prompt_state["system_prompt"],
                "output_prompt": optimized_prompt_state["output_prompt"]
            },
            model_name=model_name
        )

        # Add optimized predictions to examples
        for i, example in enumerate(examples):
            if i < len(prediction_results["predictions"]):
                example["optimized_output"] = prediction_results["predictions"][i]

        return examples
    except Exception as e:
        logger.error(f"Error in refined inference: {str(e)}")
        raise

@task(name="evaluate_refined")
def evaluate_refined(
    examples: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Step 5: Compare metrics and decide whether to continue

    Args:
        examples: List of examples with optimized predictions
        metrics: List of metrics to compute

    Returns:
        Dictionary of metric scores for optimized outputs
    """
    try:
        # Extract optimized predictions and references
        predictions = [ex.get("optimized_output", "") for ex in examples]
        references = [ex.get("ground_truth_output", "") for ex in examples]

        # Initialize evaluator
        evaluator = EvaluatorService()

        # Compute metrics for optimized outputs
        results = evaluator.evaluate(
            predictions=predictions,
            references=references,
            metrics=metrics
        )

        # Add number of examples
        results["total_examples"] = len(examples)

        return results
    except Exception as e:
        logger.error(f"Error in refined evaluation: {str(e)}")
        raise

@flow(name="prompt_optimization_flow")
def prompt_optimization_flow(
    system_prompt: str,
    output_prompt: str,
    examples: List[Dict[str, str]],
    vertex_project_id: str,
    vertex_location: str,
    primary_model_name: str,
    optimizer_model_name: str,
    metrics: Optional[List[str]] = None,
    target_threshold: float = 0.9,
    max_iterations: int = 3
) -> Dict[str, Any]:
    """
    Main flow for optimizing prompts through 5-step process

    Args:
        system_prompt: Initial system prompt text or file path
        output_prompt: Initial output prompt text or file path
        examples: List of examples for training
        vertex_project_id: Google Cloud project ID
        vertex_location: Google Cloud region
        primary_model_name: Model for inference steps
        optimizer_model_name: Model for optimization step
        metrics: List of metrics to compute
        target_threshold: Target performance threshold
        max_iterations: Maximum number of optimization iterations

    Returns:
        Dictionary with results including best prompt state and metrics
    """
    # Default metrics if not provided
    if not metrics:
        metrics = ["exact_match_score", "token_match_score"]

    # Initialize best state and metrics
    current_prompt_state = load_prompt_state(system_prompt, output_prompt)
    best_prompt_state = current_prompt_state.dict()
    best_metrics = None

    # Track history
    history = []

    # Iterate for optimization
    for iteration in range(max_iterations):
        logger.info(f"Starting iteration {iteration+1}/{max_iterations}")

        # Step 1: Primary LLM Inference
        examples_with_predictions = primary_llm_inference(
            prompt_state=current_prompt_state.dict(),
            examples=examples.copy(),  # Copy to avoid modifying original
            vertex_project_id=vertex_project_id,
            vertex_location=vertex_location,
            model_name=primary_model_name
        )

        # Step 2: Baseline Evaluation
        baseline_metrics = evaluate_baseline(
            examples=examples_with_predictions,
            metrics=metrics
        )

        # Store metrics on first iteration
        if iteration == 0:
            best_metrics = baseline_metrics.copy()

        # Step 3: Optimizer LLM
        optimized_prompt_state = optimizer_llm(
            prompt_state=current_prompt_state.dict(),
            baseline_metrics=baseline_metrics,
            example_sample=examples_with_predictions[:5],  # Sample for analysis
            vertex_project_id=vertex_project_id,
            vertex_location=vertex_location,
            optimizer_model_name=optimizer_model_name
        )

        # Step 4: Refined LLM Inference
        examples_with_optimized = refined_llm_inference(
            optimized_prompt_state=optimized_prompt_state,
            examples=examples_with_predictions,
            vertex_project_id=vertex_project_id,
            vertex_location=vertex_location,
            model_name=primary_model_name
        )

        # Step 5: Second Evaluation
        optimized_metrics = evaluate_refined(
            examples=examples_with_optimized,
            metrics=metrics
        )

        # Record this iteration
        iteration_record = {
            "iteration": iteration + 1,
            "baseline_metrics": baseline_metrics,
            "optimized_metrics": optimized_metrics,
            "baseline_prompt_state": current_prompt_state.dict(),
            "optimized_prompt_state": optimized_prompt_state,
            "examples": examples_with_optimized
        }
        history.append(iteration_record)

        # Determine if optimized prompts are better
        # Using the first metric as the primary comparison metric
        primary_metric = metrics[0]
        baseline_score = baseline_metrics.get(primary_metric, 0)
        optimized_score = optimized_metrics.get(primary_metric, 0)

        logger.info(f"Iteration {iteration+1} - Baseline {primary_metric}: {baseline_score:.4f}, Optimized: {optimized_score:.4f}")

        # Update best if optimized is better
        if optimized_score > best_metrics.get(primary_metric, 0):
            best_prompt_state = optimized_prompt_state
            best_metrics = optimized_metrics.copy()
            logger.info(f"New best prompt found with {primary_metric}: {optimized_score:.4f}")

        # Early stopping if we reach target threshold
        if optimized_score >= target_threshold:
            logger.info(f"Target threshold {target_threshold} reached. Stopping early.")
            break

        # Update current state for next iteration
        current_prompt_state = PromptState(**optimized_prompt_state)

    # Return final results
    return {
        "best_prompt_state": best_prompt_state,
        "best_metrics": best_metrics,
        "history": history,
        "iterations_completed": len(history)
    }