"""
Prefect workflow orchestration for ML operations in the ATLAS platform.

This module defines Prefect flows and tasks for:
1. Running ML experiments
2. Training and evaluating LightGBM models
3. Training and evaluating RL models
4. Executing the 5-API workflow
"""

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from typing import Dict, Any, List, Optional, Tuple
import uuid
from datetime import datetime
import json
import os
import time

from app.ml.services import MLExperimentService
from app.ml.workflow import FiveAPIWorkflow


@task(name="prepare_experiment_data")
def prepare_experiment_data(
    experiment_id: str,
    examples: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """Prepare data for an experiment.
    
    Args:
        experiment_id: ID of the experiment
        examples: List of examples to use
        
    Returns:
        Tuple of (experiment_id, prepared_examples)
    """
    logger = get_run_logger()
    logger.info(f"Preparing data for experiment {experiment_id}")
    
    # In a real implementation, this might process the examples in some way
    # For now, we're just returning them as-is
    return experiment_id, examples


@task(name="run_primary_inference")
def run_primary_inference(
    experiment_id: str,
    system_prompt: str,
    output_prompt: str,
    examples: List[Dict[str, Any]],
    llm_client=None
) -> List[Dict[str, Any]]:
    """Run primary LLM inference.
    
    Args:
        experiment_id: ID of the experiment
        system_prompt: System prompt to use
        output_prompt: Output prompt to use
        examples: List of examples to run inference on
        llm_client: Optional LLM client to use
        
    Returns:
        List of results, each containing the example and model response
    """
    logger = get_run_logger()
    logger.info(f"Running primary inference for experiment {experiment_id}")
    
    # Update experiment status
    service = MLExperimentService()
    service.update_experiment_status(experiment_id, "running")
    
    results = []
    for example in examples:
        # In a real implementation, this would call the LLM client
        if llm_client:
            response = llm_client.generate(
                system_prompt=system_prompt,
                output_prompt=output_prompt,
                user_input=example['user_input']
            )
        else:
            # Dummy response for testing
            response = f"Inference response for: {example['user_input']}"
        
        results.append({
            'example': example,
            'model_response': response
        })
    
    return results


@task(name="run_baseline_evaluation")
def run_baseline_evaluation(
    experiment_id: str,
    results: List[Dict[str, Any]],
    evaluator_client=None
) -> Dict[str, Any]:
    """Run baseline evaluation on primary inference results.
    
    Args:
        experiment_id: ID of the experiment
        results: Results from primary inference
        evaluator_client: Optional evaluator client to use
        
    Returns:
        Dictionary containing evaluation metrics
    """
    logger = get_run_logger()
    logger.info(f"Running baseline evaluation for experiment {experiment_id}")
    
    predictions = [result['model_response'] for result in results]
    references = [result['example']['ground_truth_output'] for result in results]
    
    if evaluator_client:
        metrics = evaluator_client.evaluate(predictions, references)
    else:
        # Dummy metrics for testing
        metrics = {
            'exact_match': 0.5,
            'semantic_similarity': 0.7,
            'keyword_match': 0.6,
            'bleu': 0.4
        }
    
    return metrics


@task(name="run_prompt_optimizer")
def run_prompt_optimizer(
    experiment_id: str,
    system_prompt: str,
    output_prompt: str,
    baseline_metrics: Dict[str, Any],
    examples: List[Dict[str, Any]],
    llm_client=None
) -> Dict[str, str]:
    """Run the prompt optimizer.
    
    Args:
        experiment_id: ID of the experiment
        system_prompt: Current system prompt
        output_prompt: Current output prompt
        baseline_metrics: Metrics from baseline evaluation
        examples: Training examples
        llm_client: Optional LLM client to use
        
    Returns:
        Dictionary with optimized system_prompt and output_prompt
    """
    logger = get_run_logger()
    logger.info(f"Running prompt optimizer for experiment {experiment_id}")
    
    if llm_client:
        # Example of how optimization might work with a real client
        optimization_prompt = f"""
        You are an expert prompt optimizer. Your task is to improve the following prompt pair:
        
        System Prompt: {system_prompt}
        
        Output Prompt: {output_prompt}
        
        Current metrics: {json.dumps(baseline_metrics, indent=2)}
        
        Examples:
        {json.dumps(examples[:3], indent=2)}
        
        Please provide improved versions of both prompts to achieve better performance.
        """
        
        response = llm_client.generate(
            system_prompt="You are an AI assistant that specializes in optimizing prompts.",
            output_prompt="",
            user_input=optimization_prompt
        )
        
        # In a real implementation, we would parse the response to extract the optimized prompts
        # For now, we'll just return slightly modified prompts
        optimized_system = system_prompt + " [Optimized]"
        optimized_output = output_prompt + " [Optimized]"
    else:
        # Dummy optimization for testing
        optimized_system = system_prompt + " [Optimized]"
        optimized_output = output_prompt + " [Optimized]"
    
    return {
        'system_prompt': optimized_system,
        'output_prompt': optimized_output
    }


@task(name="run_refined_inference")
def run_refined_inference(
    experiment_id: str,
    system_prompt: str,
    output_prompt: str,
    examples: List[Dict[str, Any]],
    llm_client=None
) -> List[Dict[str, Any]]:
    """Run refined LLM inference with optimized prompts.
    
    Args:
        experiment_id: ID of the experiment
        system_prompt: Optimized system prompt
        output_prompt: Optimized output prompt
        examples: List of examples to run inference on
        llm_client: Optional LLM client to use
        
    Returns:
        List of results, each containing the example and model response
    """
    logger = get_run_logger()
    logger.info(f"Running refined inference for experiment {experiment_id}")
    
    results = []
    for example in examples:
        # In a real implementation, this would call the LLM client
        if llm_client:
            response = llm_client.generate(
                system_prompt=system_prompt,
                output_prompt=output_prompt,
                user_input=example['user_input']
            )
        else:
            # Dummy response for testing that's better than the baseline
            response = f"Improved response for: {example['user_input']}"
        
        results.append({
            'example': example,
            'model_response': response
        })
    
    return results


@task(name="run_comparative_evaluation")
def run_comparative_evaluation(
    experiment_id: str,
    results: List[Dict[str, Any]],
    evaluator_client=None
) -> Dict[str, Any]:
    """Run comparative evaluation on refined inference results.
    
    Args:
        experiment_id: ID of the experiment
        results: Results from refined inference
        evaluator_client: Optional evaluator client to use
        
    Returns:
        Dictionary containing evaluation metrics
    """
    logger = get_run_logger()
    logger.info(f"Running comparative evaluation for experiment {experiment_id}")
    
    predictions = [result['model_response'] for result in results]
    references = [result['example']['ground_truth_output'] for result in results]
    
    if evaluator_client:
        metrics = evaluator_client.evaluate(predictions, references)
    else:
        # Dummy metrics for testing that show improvement
        metrics = {
            'exact_match': 0.7,
            'semantic_similarity': 0.8,
            'keyword_match': 0.75,
            'bleu': 0.6
        }
    
    return metrics


@task(name="save_experiment_results")
def save_experiment_results(
    experiment_id: str,
    iteration: int,
    system_prompt: str,
    output_prompt: str,
    baseline_metrics: Dict[str, Any],
    refined_metrics: Dict[str, Any]
) -> None:
    """Save experiment results to database.
    
    Args:
        experiment_id: ID of the experiment
        iteration: Iteration number
        system_prompt: Final system prompt
        output_prompt: Final output prompt
        baseline_metrics: Metrics from baseline evaluation
        refined_metrics: Metrics from comparative evaluation
    """
    logger = get_run_logger()
    logger.info(f"Saving results for experiment {experiment_id}, iteration {iteration}")
    
    service = MLExperimentService()
    
    # Calculate training accuracy as the average of exact_match and semantic_similarity
    training_accuracy = (
        refined_metrics.get('exact_match', 0) +
        refined_metrics.get('semantic_similarity', 0)
    ) / 2
    
    # Add the iteration to the experiment
    service.add_experiment_iteration(
        experiment_id=experiment_id,
        iteration_number=iteration,
        system_prompt=system_prompt,
        output_prompt=output_prompt,
        metrics={
            'baseline': baseline_metrics,
            'refined': refined_metrics
        },
        training_accuracy=training_accuracy,
        validation_accuracy=None  # Will be computed in a separate task
    )
    
    logger.info(f"Saved iteration {iteration} for experiment {experiment_id}")


@flow(name="5-API Workflow", 
      task_runner=SequentialTaskRunner(), 
      description="Execute the 5-API workflow for prompt optimization")
def five_api_workflow(
    experiment_id: str,
    system_prompt: str,
    output_prompt: str,
    examples: List[Dict[str, Any]],
    max_iterations: int = 5,
    target_threshold: float = 0.9,
    early_stopping_patience: int = 2
) -> Dict[str, Any]:
    """Execute the 5-API workflow as a Prefect flow.
    
    Args:
        experiment_id: ID of the experiment
        system_prompt: Initial system prompt
        output_prompt: Initial output prompt
        examples: List of examples for evaluation
        max_iterations: Maximum number of iterations
        target_threshold: Target threshold for early stopping
        early_stopping_patience: Number of iterations without improvement before stopping
        
    Returns:
        Dictionary with workflow results
    """
    logger = get_run_logger()
    logger.info(f"Starting 5-API workflow for experiment {experiment_id}")
    
    # Prepare experiment data
    _, processed_examples = prepare_experiment_data(experiment_id, examples)
    
    # Initialize state variables
    current_system_prompt = system_prompt
    current_output_prompt = output_prompt
    best_score = 0.0
    no_improvement_count = 0
    best_prompts = {
        'system_prompt': current_system_prompt,
        'output_prompt': current_output_prompt
    }
    
    # History tracking
    history = {
        'iterations': [],
        'metrics': [],
        'prompts': []
    }
    
    # Main optimization loop
    for iteration in range(1, max_iterations + 1):
        logger.info(f"Starting iteration {iteration}/{max_iterations}")
        
        # Step 1: Primary LLM Inference
        primary_results = run_primary_inference(
            experiment_id, 
            current_system_prompt, 
            current_output_prompt, 
            processed_examples
        )
        
        # Step 2: Baseline Evaluation
        baseline_metrics = run_baseline_evaluation(
            experiment_id,
            primary_results
        )
        
        # Step 3: Optimizer
        optimized_prompts = run_prompt_optimizer(
            experiment_id,
            current_system_prompt,
            current_output_prompt,
            baseline_metrics,
            processed_examples
        )
        
        # Step 4: Refined LLM Inference
        refined_results = run_refined_inference(
            experiment_id,
            optimized_prompts['system_prompt'],
            optimized_prompts['output_prompt'],
            processed_examples
        )
        
        # Step 5: Comparative Evaluation
        refined_metrics = run_comparative_evaluation(
            experiment_id,
            refined_results
        )
        
        # Save the results
        save_experiment_results(
            experiment_id,
            iteration,
            optimized_prompts['system_prompt'],
            optimized_prompts['output_prompt'],
            baseline_metrics,
            refined_metrics
        )
        
        # Calculate scores
        primary_score = (baseline_metrics.get('exact_match', 0) + baseline_metrics.get('semantic_similarity', 0)) / 2
        refined_score = (refined_metrics.get('exact_match', 0) + refined_metrics.get('semantic_similarity', 0)) / 2
        
        # Check if we have improvement
        improved = refined_score > primary_score
        
        # Store iteration data
        iteration_data = {
            'iteration': iteration,
            'primary_metrics': baseline_metrics,
            'refined_metrics': refined_metrics,
            'primary_prompts': {
                'system_prompt': current_system_prompt,
                'output_prompt': current_output_prompt
            },
            'refined_prompts': optimized_prompts,
            'improved': improved
        }
        
        # Add to history
        history['iterations'].append(iteration_data)
        history['metrics'].append({
            'iteration': iteration,
            'primary_score': primary_score,
            'refined_score': refined_score
        })
        history['prompts'].append({
            'iteration': iteration,
            'system_prompt': optimized_prompts['system_prompt'],
            'output_prompt': optimized_prompts['output_prompt']
        })
        
        # Update current prompts if improved
        if improved:
            logger.info(f"Improvement found: {primary_score:.4f} -> {refined_score:.4f}")
            current_system_prompt = optimized_prompts['system_prompt']
            current_output_prompt = optimized_prompts['output_prompt']
            
            # Update best prompts if this is the best score so far
            if refined_score > best_score:
                best_score = refined_score
                best_prompts = {
                    'system_prompt': current_system_prompt,
                    'output_prompt': current_output_prompt
                }
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        else:
            logger.info(f"No improvement: {primary_score:.4f} -> {refined_score:.4f}")
            no_improvement_count += 1
        
        # Check early stopping
        if no_improvement_count >= early_stopping_patience:
            logger.info(f"Early stopping after {iteration} iterations with no improvement")
            break
        
        # Check if we've reached the target threshold
        if refined_score >= target_threshold:
            logger.info(f"Target threshold reached: {refined_score:.4f} >= {target_threshold:.4f}")
            break
    
    # Final results
    final_results = {
        'best_prompts': best_prompts,
        'metrics': history['metrics'],
        'iterations': history['iterations'],
        'final_metrics': {
            'training_score': best_score
        }
    }
    
    # Complete the experiment
    service = MLExperimentService()
    service.complete_experiment(experiment_id, final_results)
    
    logger.info(f"5-API workflow completed for experiment {experiment_id}")
    logger.info(f"Best score (training): {best_score:.4f}")
    
    return final_results


@flow(name="Train LightGBM Model")
def train_lightgbm_model(
    model_id: str,
    experiment_ids: List[str],
    hyperparameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Train a LightGBM model on experiment results.
    
    Args:
        model_id: ID of the meta-learning model
        experiment_ids: List of experiment IDs to use for training
        hyperparameters: Dictionary of hyperparameters
        
    Returns:
        Dictionary with training results
    """
    logger = get_run_logger()
    logger.info(f"Starting LightGBM model training for model {model_id}")
    
    # Mock implementation for now
    # In a real implementation, this would:
    # 1. Load experiment data from the database
    # 2. Extract features from prompts and metrics
    # 3. Train a LightGBM model
    # 4. Save the model and results
    
    # Simulate training time
    time.sleep(2)
    
    return {
        'model_id': model_id,
        'status': 'completed',
        'metrics': {
            'train_accuracy': 0.85,
            'val_accuracy': 0.82,
            'feature_importance': {
                'prompt_length': 0.3,
                'examples_count': 0.2,
                'semantic_coherence': 0.5
            }
        }
    }


@flow(name="Train RL Model")
def train_rl_model(
    model_id: str,
    experiment_ids: List[str],
    hyperparameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Train a reinforcement learning model.
    
    Args:
        model_id: ID of the RL model
        experiment_ids: List of experiment IDs to use for training
        hyperparameters: Dictionary of hyperparameters
        
    Returns:
        Dictionary with training results
    """
    logger = get_run_logger()
    logger.info(f"Starting RL model training for model {model_id}")
    
    # Mock implementation for now
    # In a real implementation, this would:
    # 1. Load experiment data from the database
    # 2. Set up the environment and agent
    # 3. Train the RL model
    # 4. Save the model and results
    
    # Simulate training time
    time.sleep(3)
    
    return {
        'model_id': model_id,
        'status': 'completed',
        'metrics': {
            'mean_reward': 0.75,
            'episodes': 100,
            'convergence_rate': 0.92
        }
    }