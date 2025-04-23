
"""
Prefect flow for prompt optimization workflow
"""
from typing import Dict, List, Any, Optional, Tuple
import os
import json
import logging
import pandas as pd
import time
from datetime import datetime

# Import prefect if available (for production)
try:
    from prefect import flow, task, get_run_logger
    from prefect.artifacts import create_artifact
    has_prefect = True
except ImportError:
    # Fallback for development without Prefect
    has_prefect = False
    
    # Create mock decorators
    def flow(func=None, **kwargs):
        def decorator(f):
            return f
        return decorator if func is None else decorator(func)
    
    def task(func=None, **kwargs):
        def decorator(f):
            return f
        return decorator if func is None else decorator(func)
    
    def get_run_logger():
        return logging.getLogger("prefect_mock")

logger = logging.getLogger(__name__)

@task(name="load-state", retries=2)
def load_state(system_prompt_path: str, output_prompt_path: str, 
              dataset_path: str, state_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load or initialize PromptState & training data
    
    Args:
        system_prompt_path: Path to system prompt file
        output_prompt_path: Path to output prompt file
        dataset_path: Path to dataset file
        state_path: Optional path to state file
        
    Returns:
        Dictionary with state and dataset information
    """
    run_logger = get_run_logger()
    run_logger.info(f"Loading state from {system_prompt_path}, {output_prompt_path}, {dataset_path}")
    
    # Load prompts
    with open(system_prompt_path, 'r') as f:
        system_prompt = f.read()
    
    with open(output_prompt_path, 'r') as f:
        output_prompt = f.read()
    
    # Load dataset
    dataset = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_json(dataset_path)
    
    # Convert to records
    records = dataset.to_dict(orient='records')
    
    # Load or initialize state
    if state_path and os.path.exists(state_path):
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
        except Exception as e:
            run_logger.error(f"Error loading state file: {str(e)}")
            state = {
                "system_prompt": system_prompt,
                "output_prompt": output_prompt,
                "version": 1
            }
    else:
        state = {
            "system_prompt": system_prompt,
            "output_prompt": output_prompt,
            "version": 1
        }
    
    dataset_dict = {
        "data": records,
        "path": dataset_path
    }
    
    return {
        "state": state,
        "dataset": dataset_dict
    }

@task(name="vertex-primary-inference", retries=3)
def vertex_primary_inference(state_dict: Dict[str, Any], 
                            dataset_dict: Dict[str, Any], 
                            vertex_project_id: str, 
                            vertex_location: str, 
                            model_name: str) -> Dict[str, Any]:
    """
    Run primary inference with LLM (Step 1)
    
    Args:
        state_dict: Dictionary with prompt state
        dataset_dict: Dictionary with dataset information
        vertex_project_id: GCP project ID
        vertex_location: GCP location
        model_name: Model name
        
    Returns:
        Dictionary with inference results
    """
    run_logger = get_run_logger()
    run_logger.info(f"Running primary inference with model {model_name}")
    
    try:
        # Import and initialize the Vertex client
        from src.app.clients.vertex_client import VertexAIClient
        
        client = VertexAIClient(
            project_id=vertex_project_id,
            location=vertex_location
        )
        
        # Run batch prediction
        results = client.batch_predict(
            examples=dataset_dict["data"],
            prompt_state=state_dict,
            model_name=model_name
        )
        
        run_logger.info(f"Primary inference complete: {len(results['predictions'])} predictions generated")
        
        # Create artifact if using real Prefect
        if has_prefect:
            create_artifact(
                name="primary-inference-results",
                description=f"Primary inference results with {model_name}",
                data=results
            )
        
        return results
    except Exception as e:
        run_logger.error(f"Error in primary inference: {str(e)}")
        raise

@task(name="evaluate-predictions", retries=2)
def evaluate_predictions(predictions: List[str], 
                       references: List[str], 
                       metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Evaluate predictions using Hugging Face metrics (Step 2)
    
    Args:
        predictions: List of model predictions
        references: List of reference outputs
        metrics: List of metrics to compute
        
    Returns:
        Dictionary with metric scores
    """
    run_logger = get_run_logger()
    run_logger.info(f"Evaluating {len(predictions)} predictions")
    
    try:
        # Import and initialize the evaluator
        from src.app.clients.hf_evaluator import EvaluatorService
        
        evaluator = EvaluatorService()
        
        # Run evaluation
        results = evaluator.evaluate(
            predictions=predictions,
            references=references,
            metrics=metrics
        )
        
        run_logger.info(f"Evaluation complete: {results}")
        
        # Create artifact if using real Prefect
        if has_prefect:
            create_artifact(
                name="evaluation-results",
                description="Hugging Face evaluation metrics",
                data=results
            )
        
        return results
    except Exception as e:
        run_logger.error(f"Error in evaluation: {str(e)}")
        raise

@task(name="optimize-prompts", retries=2)
def optimize_prompts(state_dict: Dict[str, Any], 
                   evaluation_results: Dict[str, float], 
                   primary_results: Dict[str, Any],
                   vertex_project_id: str, 
                   vertex_location: str, 
                   optimizer_model_name: str) -> Dict[str, Any]:
    """
    Generate refined prompts using Optimizer LLM (Step 3)
    
    Args:
        state_dict: Dictionary with prompt state
        evaluation_results: Dictionary with evaluation metrics
        primary_results: Dictionary with primary inference results
        vertex_project_id: GCP project ID
        vertex_location: GCP location
        optimizer_model_name: Model name for optimizer
        
    Returns:
        Dictionary with optimized prompts and reasoning
    """
    run_logger = get_run_logger()
    run_logger.info(f"Optimizing prompts with model {optimizer_model_name}")
    
    try:
        # Import and initialize the Vertex client
        from src.app.clients.vertex_client import VertexAIClient
        
        client = VertexAIClient(
            project_id=vertex_project_id,
            location=vertex_location
        )
        
        # Prepare optimization prompt
        system_prompt = state_dict["system_prompt"]
        output_prompt = state_dict["output_prompt"]
        
        # Format metrics for the optimizer
        metrics_text = "\n".join([f"- {k}: {v}" for k, v in evaluation_results.items()])
        
        # Include sample of predictions and ground truth
        samples = []
        sample_size = min(3, len(primary_results.get("predictions", [])))
        
        for i in range(sample_size):
            try:
                prediction = primary_results["predictions"][i]
                reference = primary_results.get("references", [])[i] if "references" in primary_results else "N/A"
                user_input = primary_results.get("inputs", [])[i] if "inputs" in primary_results else "N/A"
                
                sample = f"Example #{i+1}:\nInput: {user_input}\nPrediction: {prediction}\nExpected: {reference}"
                samples.append(sample)
            except (IndexError, KeyError):
                run_logger.warning(f"Could not get sample {i}")
        
        samples_text = "\n\n".join(samples)
        
        optimizer_prompt = f"""
You are an expert prompt engineer. Your task is to improve the performance of the following prompts based on evaluation results.

## Current Prompts
### System Prompt:
{system_prompt}

### Output Prompt:
{output_prompt}

## Performance Metrics
{metrics_text}

## Sample Outputs
{samples_text}

## Your Task
Analyze the prompts, evaluation metrics, and sample outputs. Then provide:
1. A detailed analysis of issues with the current prompts
2. An improved version of the system prompt
3. An improved version of the output prompt

Format your response exactly as follows:
ANALYSIS: [Your analysis of the issues]

IMPROVED_SYSTEM_PROMPT:
[Your improved system prompt]

IMPROVED_OUTPUT_PROMPT:
[Your improved output prompt]
"""
        
        # Generate optimization
        response = client.generate_response(
            model_name=optimizer_model_name,
            user_content=optimizer_prompt,
            temperature=0.3
        )
        
        # Parse the response
        analysis = ""
        improved_system_prompt = ""
        improved_output_prompt = ""
        
        current_section = None
        
        for line in response.split("\n"):
            if line.startswith("ANALYSIS:"):
                current_section = "analysis"
                analysis = line.replace("ANALYSIS:", "").strip()
            elif line.startswith("IMPROVED_SYSTEM_PROMPT:"):
                current_section = "system"
                continue
            elif line.startswith("IMPROVED_OUTPUT_PROMPT:"):
                current_section = "output"
                continue
            elif current_section == "analysis":
                analysis += "\n" + line
            elif current_section == "system":
                improved_system_prompt += line + "\n"
            elif current_section == "output":
                improved_output_prompt += line + "\n"
        
        # Trim whitespace
        improved_system_prompt = improved_system_prompt.strip()
        improved_output_prompt = improved_output_prompt.strip()
        
        # Create new state with optimized prompts
        optimized_state = {
            "system_prompt": improved_system_prompt,
            "output_prompt": improved_output_prompt,
            "version": state_dict["version"] + 1
        }
        
        run_logger.info(f"Prompt optimization complete")
        
        # Create artifact if using real Prefect
        if has_prefect:
            create_artifact(
                name="optimization-results",
                description=f"Prompt optimization results",
                data={
                    "analysis": analysis,
                    "original_state": state_dict,
                    "optimized_state": optimized_state
                }
            )
        
        return {
            "optimized_state": optimized_state,
            "reasoning": analysis
        }
    except Exception as e:
        run_logger.error(f"Error in prompt optimization: {str(e)}")
        raise

@task(name="refined-inference", retries=3)
def refined_inference(optimized_state: Dict[str, Any], 
                    dataset_dict: Dict[str, Any], 
                    vertex_project_id: str, 
                    vertex_location: str, 
                    model_name: str) -> Dict[str, Any]:
    """
    Run inference with optimized prompts (Step 4)
    
    Args:
        optimized_state: Dictionary with optimized prompt state
        dataset_dict: Dictionary with dataset information
        vertex_project_id: GCP project ID
        vertex_location: GCP location
        model_name: Model name
        
    Returns:
        Dictionary with refined inference results
    """
    run_logger = get_run_logger()
    run_logger.info(f"Running refined inference with model {model_name}")
    
    try:
        # Import and initialize the Vertex client
        from src.app.clients.vertex_client import VertexAIClient
        
        client = VertexAIClient(
            project_id=vertex_project_id,
            location=vertex_location
        )
        
        # Run batch prediction with optimized prompts
        results = client.batch_predict(
            examples=dataset_dict["data"],
            prompt_state=optimized_state,
            model_name=model_name
        )
        
        run_logger.info(f"Refined inference complete: {len(results['predictions'])} predictions generated")
        
        # Create artifact if using real Prefect
        if has_prefect:
            create_artifact(
                name="refined-inference-results",
                description=f"Refined inference results with {model_name}",
                data=results
            )
        
        return results
    except Exception as e:
        run_logger.error(f"Error in refined inference: {str(e)}")
        raise

@task(name="compare-evaluations", retries=1)
def compare_evaluations(baseline_metrics: Dict[str, float], 
                      optimized_metrics: Dict[str, float], 
                      target_metric: str, 
                      target_threshold: float) -> Dict[str, Any]:
    """
    Compare baseline and optimized metrics (Step 5)
    
    Args:
        baseline_metrics: Dictionary with baseline metrics
        optimized_metrics: Dictionary with optimized metrics
        target_metric: Key for the target metric
        target_threshold: Target threshold for the metric
        
    Returns:
        Dictionary with comparison results
    """
    run_logger = get_run_logger()
    run_logger.info(f"Comparing metrics (target: {target_metric} >= {target_threshold})")
    
    # Get target metric values
    baseline_score = baseline_metrics.get(target_metric, 0)
    optimized_score = optimized_metrics.get(target_metric, 0)
    
    # Calculate improvement
    improvement = optimized_score - baseline_score
    percent_improvement = (improvement / baseline_score * 100) if baseline_score > 0 else 0
    
    # Check if target threshold reached
    target_achieved = optimized_score >= target_threshold
    has_improved = improvement > 0
    
    # Build comparison results
    comparison = {
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "improvement": {
            metric: optimized_metrics.get(metric, 0) - baseline_metrics.get(metric, 0)
            for metric in set(baseline_metrics.keys()) | set(optimized_metrics.keys())
        },
        "target_metric": target_metric,
        "target_metric_baseline": baseline_score,
        "target_metric_optimized": optimized_score,
        "target_metric_improvement": improvement,
        "target_metric_percent_improvement": percent_improvement,
        "target_threshold": target_threshold,
        "target_achieved": target_achieved,
        "has_improved": has_improved
    }
    
    run_logger.info(f"Comparison complete: {baseline_score} -> {optimized_score} ({percent_improvement:.2f}% improvement)")
    
    # Create artifact if using real Prefect
    if has_prefect:
        create_artifact(
            name="comparison-results",
            description=f"Comparison of baseline and optimized metrics",
            data=comparison
        )
    
    return comparison

@flow(name="prompt-optimization-flow")
def prompt_optimization_flow(
    vertex_project_id: str,
    vertex_location: str,
    primary_model_name: str,
    optimizer_model_name: str,
    dataset_path: str,
    system_prompt_path: str,
    output_prompt_path: str,
    target_metric: str = "exact_match_score",
    target_threshold: float = 0.90,
    patience: int = 3,
    max_iterations: int = 10,
) -> Dict[str, Any]:
    """
    Main optimization flow that iteratively improves prompts using Vertex AI & HF Evaluate
    
    Args:
        vertex_project_id: GCP project ID
        vertex_location: GCP location
        primary_model_name: Model name for primary inference
        optimizer_model_name: Model name for optimizer
        dataset_path: Path to dataset file
        system_prompt_path: Path to system prompt file
        output_prompt_path: Path to output prompt file
        target_metric: Key for the target metric
        target_threshold: Target threshold for the metric
        patience: Number of iterations without improvement before stopping
        max_iterations: Maximum number of iterations
        
    Returns:
        Dictionary with final results
    """
    run_logger = get_run_logger()
    run_logger.info(f"Starting prompt optimization flow (target: {target_metric} >= {target_threshold})")
    
    # Step 0: Load initial state and dataset
    initial_data = load_state(
        system_prompt_path=system_prompt_path,
        output_prompt_path=output_prompt_path,
        dataset_path=dataset_path
    )
    
    state_dict = initial_data["state"]
    dataset_dict = initial_data["dataset"]
    
    # Store artifacts and metrics for each iteration
    iterations = []
    best_score = 0
    best_state = state_dict
    patience_counter = 0
    
    # Main optimization loop
    for iteration in range(1, max_iterations + 1):
        run_logger.info(f"Starting iteration {iteration}/{max_iterations}")
        
        # Step 1: Primary LLM Inference
        primary_results = vertex_primary_inference(
            state_dict=state_dict,
            dataset_dict=dataset_dict,
            vertex_project_id=vertex_project_id,
            vertex_location=vertex_location,
            model_name=primary_model_name
        )
        
        # Step 2: Hugging Face Evaluation
        references = [example.get("ground_truth_output", "") for example in dataset_dict["data"]]
        primary_results["references"] = references
        primary_results["inputs"] = [example.get("user_input", "") for example in dataset_dict["data"]]
        
        baseline_metrics = evaluate_predictions(
            predictions=primary_results["predictions"],
            references=references
        )
        
        # Record baseline results
        baseline_results = {
            "iteration": iteration,
            "state": state_dict,
            "predictions": primary_results["predictions"],
            "metrics": baseline_metrics
        }
        
        # Check if we've already reached target threshold
        if baseline_metrics.get(target_metric, 0) >= target_threshold:
            run_logger.info(f"Target threshold reached: {baseline_metrics.get(target_metric, 0)} >= {target_threshold}")
            
            iterations.append({
                "iteration": iteration,
                "baseline": baseline_results,
                "optimized": None,
                "comparison": {
                    "target_metric": target_metric,
                    "target_threshold": target_threshold,
                    "target_achieved": True,
                    "has_improved": False,
                    "target_metric_baseline": baseline_metrics.get(target_metric, 0),
                    "target_metric_optimized": None
                },
                "reasoning": "Target threshold already reached with baseline prompts"
            })
            
            # Update best score and state
            best_score = baseline_metrics.get(target_metric, 0)
            best_state = state_dict
            
            break
        
        # Step 3: Optimizer LLM
        optimization_results = optimize_prompts(
            state_dict=state_dict,
            evaluation_results=baseline_metrics,
            primary_results=primary_results,
            vertex_project_id=vertex_project_id,
            vertex_location=vertex_location,
            optimizer_model_name=optimizer_model_name
        )
        
        optimized_state = optimization_results["optimized_state"]
        
        # Step 4: Refined LLM Inference
        refined_results = refined_inference(
            optimized_state=optimized_state,
            dataset_dict=dataset_dict,
            vertex_project_id=vertex_project_id,
            vertex_location=vertex_location,
            model_name=primary_model_name
        )
        
        refined_results["references"] = references
        refined_results["inputs"] = primary_results["inputs"]
        
        # Step 5: Second Evaluation
        optimized_metrics = evaluate_predictions(
            predictions=refined_results["predictions"],
            references=references
        )
        
        # Compare results
        comparison = compare_evaluations(
            baseline_metrics=baseline_metrics,
            optimized_metrics=optimized_metrics,
            target_metric=target_metric,
            target_threshold=target_threshold
        )
        
        # Record iteration results
        iteration_result = {
            "iteration": iteration,
            "baseline": baseline_results,
            "optimized": {
                "state": optimized_state,
                "predictions": refined_results["predictions"],
                "metrics": optimized_metrics
            },
            "comparison": comparison,
            "reasoning": optimization_results["reasoning"]
        }
        
        iterations.append(iteration_result)
        
        # Update state for next iteration if improved
        current_score = optimized_metrics.get(target_metric, 0)
        
        if current_score > best_score:
            run_logger.info(f"New best score: {current_score} (previous: {best_score})")
            best_score = current_score
            best_state = optimized_state
            patience_counter = 0
        else:
            patience_counter += 1
            run_logger.info(f"No improvement, patience: {patience_counter}/{patience}")
        
        # Check stopping conditions
        if current_score >= target_threshold:
            run_logger.info(f"Target threshold reached: {current_score} >= {target_threshold}")
            break
            
        if patience_counter >= patience:
            run_logger.info(f"Early stopping triggered after {patience} iterations without improvement")
            break
        
        # Update state for next iteration
        state_dict = optimized_state
    
    # Prepare final results
    final_results = {
        "iterations": iterations,
        "best_state": best_state,
        "best_score": best_score,
        "target_achieved": best_score >= target_threshold,
        "iterations_count": len(iterations),
        "final_system_prompt": best_state["system_prompt"],
        "final_output_prompt": best_state["output_prompt"],
        "original_system_prompt": initial_data["state"]["system_prompt"],
        "original_output_prompt": initial_data["state"]["output_prompt"]
    }
    
    run_logger.info(f"Completed prompt optimization flow with best score: {best_score}")
    
    # Create final artifact if using real Prefect
    if has_prefect:
        create_artifact(
            name="final-results",
            description=f"Final results of prompt optimization flow",
            data=final_results
        )
    
    return final_results
