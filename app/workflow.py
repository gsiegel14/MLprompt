"""
Prompt Optimization Workflow: Four-Stage API Call Sequence

This module implements the enhanced workflow for refining prompts using a four-stage
process involving:
1. Google Vertex AI - Primary LLM inference
2. Google Vertex AI - Evaluation API call 
3. Google Vertex AI - Optimizer LLM for prompt refinement
4. Hugging Face - External validation metrics

Each training run follows this sequence, with results tracked for comparison.
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
from app.huggingface_client import evaluate_metrics, compute_bleu_score, compute_exact_match, compute_rouge

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
        # Begin extensive debugging logs
        logger.info("=== STARTING NEW TRAINING CYCLE ===")
        logger.info(f"Parameters: max_iterations={max_iterations}, early_stopping_patience={early_stopping_patience}")
        logger.info(f"batch_size={batch_size}, optimizer_strategy={optimizer_strategy}, optimizer_type={optimizer_type}")
        logger.info(f"Initial system prompt length: {len(system_prompt)} chars")
        logger.info(f"Initial output prompt length: {len(output_prompt)} chars")
        
        # Start a new experiment
        experiment_id = self.experiment_tracker.start_experiment()
        logger.info(f"Started experiment {experiment_id}")
        
        # Memory debugging
        import gc
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage before training: {memory_before:.2f} MB")
        
        # Initialize variables for tracking
        current_system_prompt = system_prompt
        current_output_prompt = output_prompt
        best_score = 0.0
        no_improvement_count = 0
        best_iteration = 0
        
        # Load optimizer prompt based on type
        logger.info(f"Loading optimizer prompt type: {optimizer_type}")
        optimizer_prompt = load_optimizer_prompt(optimizer_type)
        logger.info(f"Optimizer prompt loaded, length: {len(optimizer_prompt)} chars")
        
        # Initialize the iteration counter
        final_iteration = 0
        
        # Force garbage collection before starting
        gc.collect()
        logger.info("Garbage collection performed before starting training loop")
        
        # Training loop
        for iteration in range(1, max_iterations + 1):
            final_iteration = iteration
            logger.info(f"Starting iteration {iteration} of {max_iterations}")
            
            try:
                # Memory check at start of iteration
                memory_before_iter = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory before iteration {iteration}: {memory_before_iter:.2f} MB")
                
                # ----- PHASE 1: Primary LLM Inference & Evaluation -----
                logger.info(f"PHASE 1: Starting Primary LLM Inference & Evaluation (iteration {iteration})")
                
                # Get batch of training examples
                batch = self.data_module.get_batch(batch_size=batch_size, validation=False)
                if not batch:
                    logger.warning("No training examples available")
                    return {"error": "No training examples available"}
                
                logger.info(f"Got batch of {len(batch)} examples (batch_size={batch_size})")
                
                # Process each example with the Primary LLM
                results = []
                success_count = 0
                error_count = 0
                
                # Use a smaller, more conservative batch size to prevent memory issues
                max_batch_size = 25  # Reduced from 100 to 25 examples for more reliable processing
                if batch_size == 0 or batch_size > max_batch_size:
                    logger.info(f"Limiting batch size to {max_batch_size} examples (original: {batch_size})")
                    effective_batch_size = max_batch_size
                    batch = batch[:max_batch_size] if len(batch) > max_batch_size else batch
                else:
                    effective_batch_size = batch_size
                
                # Process in very small chunks to prevent memory issues
                max_chunk_size = min(5, len(batch))  # Reduced to 5 examples per chunk for more reliable processing
                logger.info(f"Processing in chunks of {max_chunk_size} examples")
                
                # Create a log file for detailed process tracking
                from datetime import datetime
                import traceback
                
                log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                process_log_file = f"logs/process_details_{experiment_id}_{log_timestamp}.log"
                os.makedirs("logs", exist_ok=True)
                
                with open(process_log_file, 'w') as log_file:
                    log_file.write(f"=== TRAINING PROCESS LOG - Experiment {experiment_id} - {log_timestamp} ===\n")
                    log_file.write(f"System prompt length: {len(current_system_prompt)} chars\n")
                    log_file.write(f"Output prompt length: {len(current_output_prompt)} chars\n")
                    log_file.write(f"Batch size: {effective_batch_size}, Chunk size: {max_chunk_size}\n")
                    log_file.write(f"Total examples: {len(batch)}\n")
                    log_file.write("="*80 + "\n\n")
                for chunk_start in range(0, len(batch), max_chunk_size):
                    chunk_end = min(chunk_start + max_chunk_size, len(batch))
                    logger.info(f"Processing examples {chunk_start+1}-{chunk_end} of {len(batch)}")
                    
                    chunk_results = []  # Use a separate array for this chunk
                    for i in range(chunk_start, chunk_end):
                        example = batch[i]
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
                            
                            # More aggressively truncate to reduce memory usage
                            # For NEJM medical cases which can be very long
                            max_input_length = 300  # Reduced from 500 to 300
                            max_ground_truth_length = 300  # Reduced from 500 to 300
                            max_response_length = 500  # Reduced from 1000 to 500
                            
                            truncated_input = user_input[:max_input_length] + "..." if len(user_input) > max_input_length else user_input
                            truncated_ground_truth = ground_truth[:max_ground_truth_length] + "..." if len(ground_truth) > max_ground_truth_length else ground_truth
                            truncated_response = model_response[:max_response_length] + "..." if len(model_response) > max_response_length else model_response
                            
                            # Log the truncation amounts to monitor memory savings
                            if len(user_input) > max_input_length or len(ground_truth) > max_ground_truth_length or len(model_response) > max_response_length:
                                logger.info(f"Truncated example {i+1}: input {len(user_input)} → {len(truncated_input)}, " +
                                           f"ground truth {len(ground_truth)} → {len(truncated_ground_truth)}, " +
                                           f"response {len(model_response)} → {len(truncated_response)}")
                            
                            # Store the result
                            chunk_results.append({
                                'user_input': truncated_input,
                                'ground_truth_output': truncated_ground_truth,
                                'model_response': truncated_response,
                                'score': score
                            })
                            success_count += 1
                        except Exception as e:
                            logger.error(f"Error processing example {i+1}: {e}")
                            error_count += 1
                            continue
                    
                    # Add this chunk's results to the main results array
                    results.extend(chunk_results)
                    
                    # Log chunk progress to the process log
                    with open(process_log_file, 'a') as log_file:
                        log_file.write(f"CHUNK {chunk_start+1}-{chunk_end} PROCESSED\n")
                        log_file.write(f"  Success: {len(chunk_results)} examples\n")
                        log_file.write(f"  Memory before chunk: {memory_before_iter:.2f} MB\n")
                        current_memory = process.memory_info().rss / 1024 / 1024
                        log_file.write(f"  Memory after chunk: {current_memory:.2f} MB (delta: {current_memory - memory_before_iter:.2f} MB)\n")
                        log_file.write(f"  Cumulative progress: {len(results)}/{len(batch)} examples processed\n\n")
                    
                    # Clear chunk_results to free memory
                    chunk_results = []
                    
                    # Run garbage collection between chunks
                    gc.collect()
                    logger.info(f"Garbage collection performed after chunk {chunk_start+1}-{chunk_end}")
                
                logger.info(f"Completed Phase 1 with {success_count} successful examples and {error_count} errors")
                
                # Calculate overall metrics
                metrics = evaluate_batch(results)
                current_score = metrics.get('avg_score', 0)
                logger.info(f"Current average score: {current_score:.4f}")
                
                # Memory check after Phase 1
                memory_after_phase1 = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory after Phase 1: {memory_after_phase1:.2f} MB (change: {memory_after_phase1 - memory_before_iter:.2f} MB)")
                
                # ----- PHASE 2: Optimizer LLM Refinement -----
                logger.info(f"PHASE 2: Starting Optimizer LLM Refinement (iteration {iteration})")
                
                # Memory check before optimization prep
                gc.collect()  # Run garbage collection before optimization
                logger.info("Forced garbage collection before starting optimization")
                
                # Use a smaller set of examples for optimization to prevent memory issues
                max_optimization_examples = 3  # Keep small for optimization
                optimization_examples = results[:max_optimization_examples] if len(results) > max_optimization_examples else results
                logger.info(f"Using {len(optimization_examples)} examples for optimization")
                
                # Add a short pause before optimization to help avoid rate limiting
                logger.info("Pausing for 2 seconds before optimization to avoid rate limits...")
                time.sleep(2)
                
                # Make a copy of examples for the experiment tracker (up to 25 for better history display)
                max_tracker_examples = 25  # We want to save more examples for the history display
                results_for_tracker = results[:max_tracker_examples] if len(results) > max_tracker_examples else results.copy()
                logger.info(f"Saved {len(results_for_tracker)} examples for experiment tracker")
                
                # Free up original results array to save memory if it's large
                if len(results) > max_optimization_examples:
                    # Make sure we've extracted what we need before clearing
                    logger.info(f"Clearing full results array after saving {len(optimization_examples)} examples for optimization")
                    results = None
                    gc.collect()
                    logger.info("Cleared full results array to save memory")
                
                # Run additional garbage collection
                gc.collect()
                
                # Log optimization phase details to the process log
                with open(process_log_file, 'a') as log_file:
                    log_file.write("\n=== PHASE 2: OPTIMIZATION ===\n")
                    log_file.write(f"Memory before optimization: {memory_after_phase1:.2f} MB\n")
                    log_file.write(f"Examples for optimization: {len(optimization_examples)}\n")
                    log_file.write(f"Optimizer strategy: {optimizer_strategy}\n")
                    log_file.write(f"Memory after clearing full results: {process.memory_info().rss / 1024 / 1024:.2f} MB\n\n")
                
                try:
                    # Call the Optimizer LLM to improve the prompts
                    logger.info("Calling optimize_prompts function...")
                    optimization_start_time = time.time()
                    
                    optimization_result = optimize_prompts(
                        current_system_prompt,
                        current_output_prompt,
                        optimization_examples,
                        optimizer_prompt,
                        optimizer_strategy
                    )
                    
                    optimization_duration = time.time() - optimization_start_time
                    logger.info(f"optimize_prompts function completed in {optimization_duration:.2f} seconds")
                    
                    # Log optimization success to file
                    with open(process_log_file, 'a') as log_file:
                        log_file.write(f"Optimization completed successfully in {optimization_duration:.2f} seconds\n")
                        reasoning = optimization_result.get('reasoning', '')
                        reasoning_preview = reasoning[:200] + '...' if len(reasoning) > 200 else reasoning
                        log_file.write(f"Reasoning preview: {reasoning_preview}\n")
                        
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error in optimization: {error_msg}")
                    
                    # Log the error to the process log
                    with open(process_log_file, 'a') as log_file:
                        log_file.write(f"ERROR IN OPTIMIZATION: {error_msg}\n")
                        log_file.write(traceback.format_exc() + "\n")
                    
                    # Create a default result if optimization fails
                    optimization_result = {
                        "system_prompt": current_system_prompt,
                        "output_prompt": current_output_prompt,
                        "reasoning": f"Optimization failed: {error_msg}"
                    }
                
                # Memory check after Phase 2
                memory_after_phase2 = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory after Phase 2: {memory_after_phase2:.2f} MB (change: {memory_after_phase2 - memory_after_phase1:.2f} MB)")
                
                # Run garbage collection again
                gc.collect()
                logger.info("Additional garbage collection after Phase 2")
                
            except Exception as e:
                logger.error(f"Critical error in iteration {iteration}: {e}")
                # Try to recover and continue with the next iteration
                continue
            
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
                results_for_tracker,  # Using our stored subset of examples to avoid memory issues
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
            
            # Log iteration summary to the process log
            with open(process_log_file, 'a') as log_file:
                log_file.write("\n=== ITERATION SUMMARY ===\n")
                log_file.write(f"Iteration {iteration} completed\n")
                log_file.write(f"Score: {current_score:.4f}\n")
                log_file.write(f"Best score so far: {best_score:.4f} (iteration {best_iteration})\n")
                log_file.write(f"No improvement count: {no_improvement_count}/{early_stopping_patience}\n")
                
                if current_score > best_score:
                    log_file.write("IMPROVEMENT ACHIEVED - Saved as best version\n")
                else:
                    log_file.write("NO IMPROVEMENT\n")
                
                # Log memory usage
                final_memory = process.memory_info().rss / 1024 / 1024
                log_file.write(f"Final memory usage: {final_memory:.2f} MB\n")
                log_file.write(f"Memory change during iteration: {final_memory - memory_before_iter:.2f} MB\n")
                log_file.write("="*50 + "\n\n")
                
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
        # Begin extensive logging
        from datetime import datetime
        import traceback
        
        log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        validation_log_file = f"logs/validation_{log_timestamp}.log"
        os.makedirs("logs", exist_ok=True)
        
        with open(validation_log_file, 'w') as log_file:
            log_file.write(f"=== VALIDATION RUN - {log_timestamp} ===\n")
            log_file.write(f"Comparing prompt versions: {prompt_versions}\n")
            log_file.write("="*80 + "\n\n")
        
        # Memory tracking
        import gc
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage before validation: {initial_memory:.2f} MB")
        
        # Get validation examples
        validation_examples = self.data_module.get_batch(batch_size=0, validation=True)
        if not validation_examples:
            logger.warning("No validation examples available")
            with open(validation_log_file, 'a') as log_file:
                log_file.write("ERROR: No validation examples available\n")
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
            version_score = metrics.get('avg_score', 0)
            validation_results[f"v{version}"] = {
                "metrics": metrics,
                "example_count": len(results)
            }
            
            # Log validation results for this version
            with open(validation_log_file, 'a') as log_file:
                log_file.write(f"\n=== VERSION {version} VALIDATION RESULTS ===\n")
                log_file.write(f"Examples processed: {len(results)}/{len(validation_examples)}\n")
                log_file.write(f"Average score: {version_score:.4f}\n")
                log_file.write(f"Perfect matches: {metrics.get('perfect_matches', 0)}/{len(results)}\n")
                log_file.write(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB\n")
                log_file.write("="*50 + "\n")
            
            # Run garbage collection between versions
            gc.collect()
            logger.info(f"Completed validation for version {version} with score {version_score:.4f}")
            
            # Short pause to avoid API rate limits
            time.sleep(1)
        
        # Create and log a final summary
        with open(validation_log_file, 'a') as log_file:
            log_file.write("\n=== VALIDATION SUMMARY ===\n")
            log_file.write(f"Total versions compared: {len(validation_results)}\n")
            log_file.write(f"Total examples processed: {len(validation_examples)}\n")
            
            # Find the best performing version
            best_version = None
            best_score = 0
            for version, results in validation_results.items():
                version_score = results.get('metrics', {}).get('avg_score', 0)
                if version_score > best_score:
                    best_score = version_score
                    best_version = version
            
            if best_version:
                log_file.write(f"Best performing version: {best_version} (score: {best_score:.4f})\n")
                
                # List all versions and their scores
                log_file.write("\nAll version scores:\n")
                for version, results in validation_results.items():
                    version_score = results.get('metrics', {}).get('avg_score', 0)
                    perfect_matches = results.get('metrics', {}).get('perfect_matches', 0)
                    example_count = results.get('example_count', 0)
                    log_file.write(f"  {version}: {version_score:.4f} ({perfect_matches}/{example_count} perfect matches)\n")
            else:
                log_file.write("No valid results to compare\n")
                
            # Final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024
            log_file.write(f"\nMemory usage: {final_memory:.2f} MB (change: {final_memory - initial_memory:.2f} MB)\n")
            log_file.write(f"Validation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write("="*80 + "\n")
            
        logger.info(f"Validation completed and logged to {validation_log_file}")
        
        # Run a final garbage collection
        gc.collect()
        
        return {
            "validation_results": validation_results,
            "example_count": len(validation_examples),
            "log_file": validation_log_file  # Return the log file path for reference
        }