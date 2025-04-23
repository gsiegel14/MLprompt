"""
Prompt Optimization Flow using Prefect 2.0
"""
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from prefect import flow, task, get_run_logger
from typing import Dict, List, Any, Optional

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
    best_examples = None

    # Track history
    history = []

    #Simple cost tracker (replace with a real implementation)
    class SimpleCostTracker:
        def __init__(self):
            self.costs = []

        def record_cost(self, cost):
            self.costs.append(cost)

        def get_cost_report(self):
            return {"total_cost": sum(self.costs)}

        def save_report(self, filename):
            import json
            with open(filename, 'w') as f:
                json.dump(self.get_cost_report(), f)
            return filename

    cost_tracker = SimpleCostTracker()

    optimization_start_time = time.time()

    # Iterate for optimization
    for iteration in range(max_iterations):
        logger.info(f"Starting iteration {iteration+1}/{max_iterations}")
        #Record cost for this iteration (replace with actual cost calculation)
        cost_tracker.record_cost(iteration + 1) #Example cost

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
            best_examples = examples_with_predictions.copy()

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
            best_examples = examples_with_optimized.copy()
            logger.info(f"New best prompt found with {primary_metric}: {optimized_score:.4f}")

        # Early stopping if we reach target threshold
        if optimized_score >= target_threshold:
            logger.info(f"Target threshold {target_threshold} reached. Stopping early.")
            break

        # Update current state for next iteration
        current_prompt_state = PromptState(**optimized_prompt_state)

    # Calculate total optimization time
    optimization_duration = time.time() - optimization_start_time

    # Save final cost report
    cost_report_path = cost_tracker.save_report("final_optimization_costs.json")

    # Return the best prompt state, metrics, and cost information
    return {
        "best_prompt_state": best_prompt_state,
        "best_metrics": best_metrics,
        "history": history,
        "iterations_completed": len(history),
        "optimization_duration_seconds": optimization_duration,
        "cost_report": cost_tracker.get_cost_report(),
        "cost_report_path": cost_report_path
    }
"""
Prefect flow for the 5-step prompt optimization workflow
"""
from prefect import flow, get_run_logger
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.flows.tasks.data_tasks import load_state, save_state
from src.flows.tasks.inference_tasks import vertex_primary_inference, vertex_refined_inference
from src.flows.tasks.evaluation_tasks import hf_eval_baseline, hf_eval_refined
from src.flows.tasks.optimization_tasks import vertex_optimizer_refine
from src.flows.tasks.logging_tasks import compare_and_log
from src.app.config import settings
from src.app.utils import calculate_score

@flow(name="prompt-optimization-flow")
def prompt_optimization_flow(
    system_prompt_path: str,
    output_prompt_path: str,
    dataset_path: str,
    metric_names: List[str] = None,
    target_metric: str = "avg_score",
    target_threshold: float = 0.90,
    patience: int = 3,
    max_iterations: int = 10,
    batch_size: int = 10,
    sample_k: int = 5,
    optimizer_strategy: str = "reasoning_first",
    experiment_id: str = None,
    state_path: str = None,
):
    """
    5-step workflow for prompt optimization:
    1. Primary LLM Inference
    2. Hugging Face Evaluation
    3. Optimizer LLM
    4. Refined LLM Inference
    5. Second Evaluation
    
    Args:
        system_prompt_path: Path to initial system prompt file
        output_prompt_path: Path to initial output prompt file
        dataset_path: Path to dataset file (CSV or JSON)
        metric_names: List of metrics to calculate
        target_metric: Primary metric for optimization
        target_threshold: Target value for early stopping
        patience: Number of non-improving iterations before stopping
        max_iterations: Maximum number of optimization cycles
        batch_size: Batch size for API calls
        sample_k: Number of worst examples to send to optimizer
        optimizer_strategy: Strategy for optimization
        experiment_id: Experiment ID to track in experiment history
        state_path: Optional path to existing state to resume
    
    Returns:
        Dictionary with final results
    """
    logger = get_run_logger()
    logger.info(f"Starting prompt optimization flow with target {target_metric} >= {target_threshold}")
    
    # Set default metric names if not provided
    if metric_names is None:
        metric_names = ["exact_match", "bleu"]
    
    # Initialize experiment tracking
    if experiment_id is None:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment directory
    experiment_dir = os.path.join("experiments", experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize tracking variables
    no_improve_count = 0
    best_metric_value = 0.0
    best_state_path = None
    
    # Track all results
    results_history = []
    
    # Start the optimization loop
    for iteration in range(max_iterations):
        logger.info(f"Starting iteration {iteration+1}/{max_iterations}")
        
        # 1. Load or initialize state and dataset
        state_data = load_state(
            system_prompt_path, 
            output_prompt_path, 
            dataset_path, 
            state_path
        )
        
        # 2. Run primary inference (Step 1)
        dataset_with_preds = vertex_primary_inference(
            state_data["prompt_state"],
            state_data["dataset"],
            batch_size=batch_size
        )
        
        # 3. Evaluate baseline performance (Step 2)
        baseline_result = hf_eval_baseline(
            dataset_with_preds,
            metric_names
        )
        baseline_metrics = baseline_result["metrics"]
        dataset_with_scores = baseline_result["dataset"]
        
        # Log baseline metrics
        logger.info(f"Baseline metrics: {json.dumps(baseline_metrics)}")
        
        # 4. Generate refined prompts (Step 3)
        refined_prompt_state = vertex_optimizer_refine(
            state_data["prompt_state"],
            dataset_with_scores,
            baseline_metrics,
            optimizer_strategy=optimizer_strategy,
            sample_k=sample_k
        )
        
        # 5. Run inference with refined prompts (Step 4)
        refined_dataset = vertex_refined_inference(
            refined_prompt_state,
            dataset_with_scores,
            batch_size=batch_size
        )
        
        # 6. Evaluate refined performance (Step 5)
        refined_result = hf_eval_refined(
            refined_dataset,
            metric_names
        )
        refined_metrics = refined_result["metrics"]
        
        # Log refined metrics
        logger.info(f"Refined metrics: {json.dumps(refined_metrics)}")
        
        # 7. Compare and decide whether to continue
        decision = compare_and_log(
            baseline_metrics,
            refined_metrics,
            state_data["prompt_state"],
            refined_prompt_state,
            iteration+1,
            target_metric,
            target_threshold,
            patience,
            no_improve_count
        )
        
        # Update control variables
        no_improve_count = decision["no_improve_count"]
        
        # Save iteration data
        iteration_dir = os.path.join(experiment_dir, f"iteration_{iteration+1}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Save prompts
        with open(os.path.join(iteration_dir, "original_system.txt"), "w") as f:
            f.write(state_data["prompt_state"]["system_prompt"])
        with open(os.path.join(iteration_dir, "original_output.txt"), "w") as f:
            f.write(state_data["prompt_state"]["output_prompt"])
        with open(os.path.join(iteration_dir, "refined_system.txt"), "w") as f:
            f.write(refined_prompt_state["system_prompt"])
        with open(os.path.join(iteration_dir, "refined_output.txt"), "w") as f:
            f.write(refined_prompt_state["output_prompt"])
        
        # Save metrics
        with open(os.path.join(iteration_dir, "metrics.json"), "w") as f:
            json.dump({
                "baseline": baseline_metrics,
                "refined": refined_metrics,
                "decision": decision
            }, f, indent=2)
        
        # Save a sample of examples with both responses
        example_sample = [refined_dataset[i] for i in range(min(5, len(refined_dataset)))]
        with open(os.path.join(iteration_dir, "examples.json"), "w") as f:
            json.dump(example_sample, f, indent=2)
        
        # Determine which state to save for the next iteration
        state_to_use = refined_prompt_state if decision["use_refined"] else state_data["prompt_state"]
        state_path = save_state(state_to_use, iteration+1)
        
        # Track best state
        current_value = decision["target_value"]
        if current_value > best_metric_value:
            best_metric_value = current_value
            best_state_path = state_path
            
            # Save best prompts
            if decision["use_refined"]:
                with open(os.path.join(experiment_dir, "best_system_prompt.txt"), "w") as f:
                    f.write(refined_prompt_state["system_prompt"])
                with open(os.path.join(experiment_dir, "best_output_prompt.txt"), "w") as f:
                    f.write(refined_prompt_state["output_prompt"])
            else:
                with open(os.path.join(experiment_dir, "best_system_prompt.txt"), "w") as f:
                    f.write(state_data["prompt_state"]["system_prompt"])
                with open(os.path.join(experiment_dir, "best_output_prompt.txt"), "w") as f:
                    f.write(state_data["prompt_state"]["output_prompt"])
        
        # Add to results history
        results_history.append({
            "iteration": iteration+1,
            "baseline_metrics": baseline_metrics,
            "refined_metrics": refined_metrics,
            "decision": decision,
            "state_path": state_path
        })
        
        # Check if we should stop early
        if decision["should_stop"]:
            logger.info(f"Early stopping at iteration {iteration+1}")
            break
    
    # Create final summary
    final_results = {
        "experiment_id": experiment_id,
        "iterations_completed": iteration+1,
        "best_metric_value": best_metric_value,
        "best_state_path": best_state_path,
        "history": results_history,
        "final_metrics": refined_metrics if decision.get("use_refined", False) else baseline_metrics
    }
    
    # Save final results
    with open(os.path.join(experiment_dir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"Optimization completed with best {target_metric}: {best_metric_value:.4f}")
    return final_results
