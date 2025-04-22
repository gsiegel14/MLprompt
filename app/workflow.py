"""
Prompt Optimization Workflow: Two-Stage Training Cycle

This module implements the specific workflow for refining prompts using a two-stage 
process involving a Primary LLM and an Optimizer LLM within each training run, 
followed by a separate validation step.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
import time

from app.data_module import DataModule
from app.evaluator import calculate_score, evaluate_batch
from app.llm_client import get_llm_response
from app.optimizer import optimize_prompts, load_optimizer_prompt
from app.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)

class PromptOptimizationWorkflow:
    """
    Implements the Two-Stage Training Cycle workflow for prompt optimization.
    
    Workflow:
    1. Primary LLM Inference & Evaluation (Phase 1)
       - Process training data with current prompts
       - Evaluate responses against ground truth
    2. Optimizer LLM Refinement (Phase 2)
       - Analyze results and generate improved prompts
    3. Validation (Separate phase)
       - Compare different prompt versions on unseen data
    """
    
    def __init__(self, data_module: DataModule, experiment_tracker: ExperimentTracker, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the workflow.
        
        Args:
            data_module: The DataModule instance for handling datasets
            experiment_tracker: The ExperimentTracker for recording results
            config: Configuration for the workflow
        """
        self.data_module = data_module
        self.experiment_tracker = experiment_tracker
        self.config = config or {}
        
        # Ensure we have the optimizer directory
        os.makedirs(os.path.join('prompts', 'optimizer'), exist_ok=True)
        
    def run_training_cycle(self, 
                          system_prompt: str, 
                          output_prompt: str, 
                          max_iterations: int = 1,
                          early_stopping_patience: int = 2,
                          batch_size: int = 0,
                          optimizer_strategy: str = 'full_rewrite',
                          optimizer_type: str = 'general') -> Dict[str, Any]:
        """
        Run a full training cycle with the specified number of iterations.
        
        Args:
            system_prompt: The initial system prompt
            output_prompt: The initial output prompt
            max_iterations: Maximum number of training cycles
            early_stopping_patience: Stop if no improvement after this many iterations
            batch_size: Number of examples per batch (0 for all)
            optimizer_strategy: Strategy for the optimizer ('full_rewrite', 'targeted_edit', etc.)
            optimizer_type: Type of optimizer ('general', 'medical')
            
        Returns:
            dict: Results of the training cycle
        """
        # Start a new experiment
        experiment_id = self.experiment_tracker.start_experiment()
        logger.info(f"Started experiment {experiment_id}")
        
        # Initialize variables for tracking
        current_system_prompt = system_prompt
        current_output_prompt = output_prompt
        best_score = 0.0
        no_improvement_count = 0
        best_iteration = 0
        
        # Load optimizer prompt based on type
        optimizer_prompt = load_optimizer_prompt(optimizer_type)
        
        # Initialize the iteration counter
        final_iteration = 0
        
        # Training loop
        for iteration in range(1, max_iterations + 1):
            final_iteration = iteration
            logger.info(f"Starting iteration {iteration} of {max_iterations}")
            
            # ----- PHASE 1: Primary LLM Inference & Evaluation -----
            
            # Get batch of training examples
            batch = self.data_module.get_batch(batch_size=batch_size, validation=False)
            if not batch:
                logger.warning("No training examples available")
                return {"error": "No training examples available"}
            
            # Process each example with the Primary LLM
            results = []
            for i, example in enumerate(batch):
                try:
                    user_input = example.get('user_input', '')
                    ground_truth = example.get('ground_truth_output', '')
                    
                    # Call the Primary LLM
                    logger.info(f"Processing example {i+1}/{len(batch)}")
                    model_response = get_llm_response(
                        current_system_prompt,
                        user_input,
                        current_output_prompt,
                        self.config.get('gemini', {})
                    )
                    
                    # Evaluate the response
                    score = calculate_score(model_response, ground_truth)
                    
                    # Store the result
                    results.append({
                        'user_input': user_input,
                        'ground_truth_output': ground_truth,
                        'model_response': model_response,
                        'score': score
                    })
                except Exception as e:
                    logger.error(f"Error processing example {i}: {e}")
                    continue
            
            # Calculate overall metrics
            metrics = evaluate_batch(results)
            current_score = metrics.get('avg_score', 0)
            
            # ----- PHASE 2: Optimizer LLM Refinement -----
            
            # Call the Optimizer LLM to improve the prompts
            optimization_result = optimize_prompts(
                current_system_prompt,
                current_output_prompt,
                results,
                optimizer_prompt,
                optimizer_strategy
            )
            
            # Extract the new prompts
            new_system_prompt = optimization_result.get('system_prompt', current_system_prompt)
            new_output_prompt = optimization_result.get('output_prompt', current_output_prompt)
            optimizer_reasoning = optimization_result.get('reasoning', '')
            
            # Save the results of this iteration
            self.experiment_tracker.save_iteration(
                experiment_id,
                iteration,
                current_system_prompt,
                current_output_prompt,
                metrics,
                results[:5],  # Include only a few examples to avoid excessive storage
                optimizer_reasoning
            )
            
            # Update the current prompts for the next iteration
            current_system_prompt = new_system_prompt
            current_output_prompt = new_output_prompt
            
            # Check for improvement
            if current_score > best_score:
                best_score = current_score
                best_iteration = iteration
                no_improvement_count = 0
                
                # Save the best prompts
                with open(os.path.join('prompts', f'system_v{iteration}.txt'), 'w') as f:
                    f.write(current_system_prompt)
                with open(os.path.join('prompts', f'output_v{iteration}.txt'), 'w') as f:
                    f.write(current_output_prompt)
            else:
                no_improvement_count += 1
                
            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                logger.info(f"Early stopping after {iteration} iterations (no improvement for {no_improvement_count} iterations)")
                break
                
            # Short pause to avoid API rate limits
            time.sleep(1)
        
        # Return the final results
        return {
            "experiment_id": experiment_id,
            "iterations": final_iteration,
            "best_iteration": best_iteration,
            "best_score": best_score,
            "final_system_prompt": current_system_prompt,
            "final_output_prompt": current_output_prompt
        }
    
    def run_validation(self, prompt_versions: List[int]) -> Dict[str, Any]:
        """
        Run validation to compare different prompt versions.
        
        Args:
            prompt_versions: List of version numbers to compare
            
        Returns:
            dict: Validation results
        """
        # Get validation examples
        validation_examples = self.data_module.get_batch(batch_size=0, validation=True)
        if not validation_examples:
            logger.warning("No validation examples available")
            return {"error": "No validation examples available"}
        
        # Collect results for each prompt version
        validation_results = {}
        
        for version in prompt_versions:
            # Load the prompts for this version
            system_prompt_path = os.path.join('prompts', f'system_v{version}.txt')
            output_prompt_path = os.path.join('prompts', f'output_v{version}.txt')
            
            if not (os.path.exists(system_prompt_path) and os.path.exists(output_prompt_path)):
                logger.warning(f"Prompt version {version} not found")
                continue
                
            with open(system_prompt_path, 'r') as f:
                system_prompt = f.read()
            with open(output_prompt_path, 'r') as f:
                output_prompt = f.read()
                
            # Process validation examples
            results = []
            for i, example in enumerate(validation_examples):
                try:
                    user_input = example.get('user_input', '')
                    ground_truth = example.get('ground_truth_output', '')
                    
                    # Call the Primary LLM
                    model_response = get_llm_response(
                        system_prompt,
                        user_input,
                        output_prompt,
                        self.config.get('gemini', {})
                    )
                    
                    # Evaluate the response
                    score = calculate_score(model_response, ground_truth)
                    
                    # Store the result
                    results.append({
                        'user_input': user_input,
                        'ground_truth_output': ground_truth,
                        'model_response': model_response,
                        'score': score
                    })
                except Exception as e:
                    logger.error(f"Error processing validation example {i} with version {version}: {e}")
                    continue
            
            # Calculate metrics for this version
            metrics = evaluate_batch(results)
            validation_results[f"v{version}"] = {
                "metrics": metrics,
                "example_count": len(results)
            }
            
            # Short pause to avoid API rate limits
            time.sleep(1)
        
        return {
            "validation_results": validation_results,
            "example_count": len(validation_examples)
        }