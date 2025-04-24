"""
ML workflow orchestration for the ATLAS platform.

This module implements the 5-API workflow using a pipeline architecture:
1. Primary LLM Inference (Vertex AI)
2. Baseline Evaluation (Hugging Face)
3. Optimizer (Vertex AI)
4. Refined LLM Inference (Vertex AI)
5. Comparative Evaluation (Hugging Face)
"""

import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime

from app.ml.services import MLExperimentService
from app import db
from app.ml.models import ModelConfiguration, MetricConfiguration, MLExperiment

logger = logging.getLogger(__name__)

class FiveAPIWorkflow:
    """
    Implementation of the 5-API workflow for prompt optimization.
    
    This class orchestrates the full workflow involving:
    1. Primary LLM Inference - Generate baseline responses (Vertex AI)
    2. Baseline Evaluation - Compute metrics (Hugging Face)
    3. Optimizer - Generate refined prompts (Vertex AI)
    4. Refined LLM Inference - Generate new responses (Vertex AI)
    5. Comparative Evaluation - Final metrics (Hugging Face)
    """
    
    def __init__(
        self,
        experiment_id: str,
        llm_client=None,
        evaluator_client=None,
        data_module=None,
        max_iterations: int = 5,
        target_metric: str = "exact_match",
        target_threshold: float = 0.9,
        early_stopping_patience: int = 2
    ):
        """Initialize the workflow.
        
        Args:
            experiment_id: ID of the experiment to track progress
            llm_client: Client for LLM API calls
            evaluator_client: Client for evaluation API calls
            data_module: Data access module for examples
            max_iterations: Maximum number of optimization iterations
            target_metric: Primary metric to optimize for
            target_threshold: Target threshold for the primary metric
            early_stopping_patience: Number of iterations without improvement before stopping
        """
        self.experiment_id = experiment_id
        self.llm_client = llm_client
        self.evaluator_client = evaluator_client
        self.data_module = data_module
        self.max_iterations = max_iterations
        self.target_metric = target_metric
        self.target_threshold = target_threshold
        self.early_stopping_patience = early_stopping_patience
        
        # State variables
        self.experiment_service = MLExperimentService()
        self.experiment = self._load_experiment()
        self.current_iteration = 0
        self.best_score = 0.0
        self.no_improvement_count = 0
        
        # Load configurations
        self.model_config = self._get_model_config()
        self.metric_config = self._get_metric_config()
        
        # Tracking history
        self.history = {
            'iterations': [],
            'metrics': [],
            'prompts': []
        }
    
    def _load_experiment(self) -> Dict[str, Any]:
        """Load the experiment details.
        
        Returns:
            Dictionary containing experiment details
        """
        experiment_dict = self.experiment_service.get_experiment(self.experiment_id)
        if not experiment_dict:
            raise ValueError(f"Experiment with ID {self.experiment_id} not found")
        return experiment_dict
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get the model configuration for this experiment.
        
        Returns:
            Dictionary containing model configuration
        """
        model_config_id = self.experiment.get('model_config_id')
        if not model_config_id:
            raise ValueError(f"Experiment {self.experiment_id} has no model configuration")
        
        config = ModelConfiguration.query.filter_by(id=model_config_id).first()
        if not config:
            raise ValueError(f"Model configuration {model_config_id} not found")
        
        return config.to_dict()
    
    def _get_metric_config(self) -> Dict[str, Any]:
        """Get the metric configuration for this experiment.
        
        Returns:
            Dictionary containing metric configuration
        """
        metric_config_id = self.experiment.get('metric_config_id')
        if not metric_config_id:
            raise ValueError(f"Experiment {self.experiment_id} has no metric configuration")
        
        config = MetricConfiguration.query.filter_by(id=metric_config_id).first()
        if not config:
            raise ValueError(f"Metric configuration {metric_config_id} not found")
        
        return config.to_dict()
    
    def run(self, system_prompt: str, output_prompt: str, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the full 5-API workflow.
        
        Args:
            system_prompt: Initial system prompt to optimize
            output_prompt: Initial output prompt to optimize
            examples: List of examples for evaluation (each with user_input and ground_truth_output)
            
        Returns:
            Dictionary with workflow results including:
            - best_prompts: The best-performing prompts
            - metrics: Metrics for each iteration
            - final_metrics: Metrics for the final prompts
        """
        logger.info(f"Starting 5-API workflow for experiment {self.experiment_id}")
        
        # Update experiment status
        self.experiment_service.update_experiment_status(self.experiment_id, "running")
        
        current_system_prompt = system_prompt
        current_output_prompt = output_prompt
        
        # Split examples into training and validation
        train_examples, validation_examples = self._split_examples(examples)
        
        # Track the best prompts
        best_prompts = {
            'system_prompt': current_system_prompt,
            'output_prompt': current_output_prompt
        }
        
        # Main optimization loop
        for iteration in range(1, self.max_iterations + 1):
            self.current_iteration = iteration
            logger.info(f"Starting iteration {iteration}/{self.max_iterations}")
            
            # Step 1: Primary LLM Inference
            logger.info("Step 1: Running primary LLM inference")
            primary_results = self._run_primary_inference(
                current_system_prompt, 
                current_output_prompt, 
                train_examples
            )
            
            # Step 2: Baseline Evaluation
            logger.info("Step 2: Running baseline evaluation")
            baseline_metrics = self._run_baseline_evaluation(
                primary_results,
                train_examples
            )
            
            # Step 3: Optimizer
            logger.info("Step 3: Running optimizer")
            optimized_prompts = self._run_optimizer(
                current_system_prompt,
                current_output_prompt,
                baseline_metrics,
                train_examples
            )
            
            # Step 4: Refined LLM Inference
            logger.info("Step 4: Running refined LLM inference")
            refined_results = self._run_refined_inference(
                optimized_prompts['system_prompt'],
                optimized_prompts['output_prompt'],
                train_examples
            )
            
            # Step 5: Comparative Evaluation
            logger.info("Step 5: Running comparative evaluation")
            refined_metrics = self._run_comparative_evaluation(
                refined_results,
                train_examples
            )
            
            # Check if we have improvement
            improved, primary_score, refined_score = self._check_improvement(baseline_metrics, refined_metrics)
            
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
            self.history['iterations'].append(iteration_data)
            self.history['metrics'].append({
                'iteration': iteration,
                'primary_score': primary_score,
                'refined_score': refined_score
            })
            self.history['prompts'].append({
                'iteration': iteration,
                'system_prompt': optimized_prompts['system_prompt'],
                'output_prompt': optimized_prompts['output_prompt']
            })
            
            # Save iteration to database
            self._save_iteration(
                iteration,
                optimized_prompts['system_prompt'],
                optimized_prompts['output_prompt'],
                {
                    'primary_score': primary_score,
                    'refined_score': refined_score,
                    'primary_metrics': baseline_metrics,
                    'refined_metrics': refined_metrics
                },
                training_accuracy=refined_score,
                validation_accuracy=None  # Will be computed later
            )
            
            # Update current prompts if improved
            if improved:
                logger.info(f"Improvement found: {primary_score:.4f} -> {refined_score:.4f}")
                current_system_prompt = optimized_prompts['system_prompt']
                current_output_prompt = optimized_prompts['output_prompt']
                
                # Update best prompts if this is the best score so far
                if refined_score > self.best_score:
                    self.best_score = refined_score
                    best_prompts = {
                        'system_prompt': current_system_prompt,
                        'output_prompt': current_output_prompt
                    }
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
            else:
                logger.info(f"No improvement: {primary_score:.4f} -> {refined_score:.4f}")
                self.no_improvement_count += 1
            
            # Check early stopping
            if self.no_improvement_count >= self.early_stopping_patience:
                logger.info(f"Early stopping after {iteration} iterations with no improvement")
                break
            
            # Check if we've reached the target threshold
            if refined_score >= self.target_threshold:
                logger.info(f"Target threshold reached: {refined_score:.4f} >= {self.target_threshold:.4f}")
                break
        
        # Final validation on best prompts
        logger.info("Running final validation on best prompts")
        validation_results = self._run_refined_inference(
            best_prompts['system_prompt'],
            best_prompts['output_prompt'],
            validation_examples
        )
        
        validation_metrics = self._run_comparative_evaluation(
            validation_results,
            validation_examples
        )
        
        validation_score = self._calculate_score(validation_metrics)
        
        # Final results
        final_results = {
            'best_prompts': best_prompts,
            'metrics': self.history['metrics'],
            'iterations': self.history['iterations'],
            'final_metrics': {
                'training_score': self.best_score,
                'validation_score': validation_score,
                'training_metrics': refined_metrics,
                'validation_metrics': validation_metrics
            }
        }
        
        # Complete the experiment
        self.experiment_service.complete_experiment(self.experiment_id, final_results)
        
        logger.info(f"5-API workflow completed for experiment {self.experiment_id}")
        logger.info(f"Best score (training): {self.best_score:.4f}")
        logger.info(f"Validation score: {validation_score:.4f}")
        
        return final_results
    
    def _split_examples(self, examples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split examples into training and validation sets.
        
        Args:
            examples: List of examples to split
            
        Returns:
            Tuple of (train_examples, validation_examples)
        """
        # For now, we use a simple 80/20 split
        split_idx = int(len(examples) * 0.8)
        return examples[:split_idx], examples[split_idx:]
    
    def _run_primary_inference(
        self, 
        system_prompt: str, 
        output_prompt: str, 
        examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run the primary LLM inference (Step 1).
        
        Args:
            system_prompt: System prompt to use
            output_prompt: Output prompt to use
            examples: List of examples to run inference on
            
        Returns:
            List of results, each containing the example and model response
        """
        # TODO: Implement actual LLM inference
        # For now, we'll return dummy results
        results = []
        for example in examples:
            # In a real implementation, this would call the LLM client
            if self.llm_client:
                response = self.llm_client.generate(
                    system_prompt=system_prompt,
                    output_prompt=output_prompt,
                    user_input=example['user_input']
                )
            else:
                # Dummy response for testing
                response = f"Dummy response for: {example['user_input']}"
            
            results.append({
                'example': example,
                'model_response': response
            })
        
        return results
    
    def _run_baseline_evaluation(
        self, 
        results: List[Dict[str, Any]], 
        examples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run the baseline evaluation (Step 2).
        
        Args:
            results: Results from primary inference
            examples: Original examples with ground truth
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # TODO: Implement actual evaluation using Hugging Face
        # For now, we'll return dummy metrics
        predictions = [result['model_response'] for result in results]
        references = [example['ground_truth_output'] for example in examples]
        
        if self.evaluator_client:
            metrics = self.evaluator_client.evaluate(predictions, references)
        else:
            # Dummy metrics for testing
            metrics = {
                'exact_match': 0.5,
                'semantic_similarity': 0.7,
                'keyword_match': 0.6,
                'bleu': 0.4
            }
        
        return metrics
    
    def _run_optimizer(
        self, 
        system_prompt: str, 
        output_prompt: str, 
        baseline_metrics: Dict[str, Any],
        examples: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Run the optimizer (Step 3).
        
        Args:
            system_prompt: Current system prompt
            output_prompt: Current output prompt
            baseline_metrics: Metrics from baseline evaluation
            examples: Training examples
            
        Returns:
            Dictionary with optimized system_prompt and output_prompt
        """
        # TODO: Implement actual optimizer using LLM
        # For now, we'll return slightly modified prompts
        if self.llm_client:
            # Example of how optimization might work with a real client
            optimization_prompt = f"""
            You are a prompt optimization expert. Your task is to improve the following prompts
            based on evaluation results.
            
            CURRENT SYSTEM PROMPT:
            {system_prompt}
            
            CURRENT OUTPUT PROMPT:
            {output_prompt}
            
            EVALUATION METRICS:
            {json.dumps(baseline_metrics, indent=2)}
            
            Based on these metrics, provide improved versions of both prompts.
            Format your response as a JSON with "system_prompt" and "output_prompt" keys.
            """
            
            optimization_result = self.llm_client.optimize(optimization_prompt)
            try:
                optimized_prompts = json.loads(optimization_result)
                return {
                    'system_prompt': optimized_prompts.get('system_prompt', system_prompt),
                    'output_prompt': optimized_prompts.get('output_prompt', output_prompt)
                }
            except:
                logger.error("Failed to parse optimization result as JSON")
                return {
                    'system_prompt': system_prompt,
                    'output_prompt': output_prompt
                }
        else:
            # Dummy optimization for testing
            optimized_system_prompt = system_prompt + " Be more precise and detailed in your responses."
            optimized_output_prompt = output_prompt + " Ensure your answer is comprehensive and accurate."
            
            return {
                'system_prompt': optimized_system_prompt,
                'output_prompt': optimized_output_prompt
            }
    
    def _run_refined_inference(
        self, 
        system_prompt: str, 
        output_prompt: str, 
        examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run the refined LLM inference (Step 4).
        
        Args:
            system_prompt: Optimized system prompt
            output_prompt: Optimized output prompt
            examples: List of examples to run inference on
            
        Returns:
            List of results, each containing the example and model response
        """
        # This is similar to _run_primary_inference but uses the optimized prompts
        # In a real implementation, this might use different parameters
        return self._run_primary_inference(system_prompt, output_prompt, examples)
    
    def _run_comparative_evaluation(
        self, 
        results: List[Dict[str, Any]], 
        examples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run the comparative evaluation (Step 5).
        
        Args:
            results: Results from refined inference
            examples: Original examples with ground truth
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Similar to _run_baseline_evaluation
        return self._run_baseline_evaluation(results, examples)
    
    def _check_improvement(
        self, 
        baseline_metrics: Dict[str, Any], 
        refined_metrics: Dict[str, Any]
    ) -> Tuple[bool, float, float]:
        """Check if there is improvement in the metrics.
        
        Args:
            baseline_metrics: Metrics from baseline evaluation
            refined_metrics: Metrics from refined evaluation
            
        Returns:
            Tuple of (improved, baseline_score, refined_score)
        """
        baseline_score = self._calculate_score(baseline_metrics)
        refined_score = self._calculate_score(refined_metrics)
        
        # Allow a small tolerance for noise
        improvement_threshold = 0.001
        improved = refined_score > (baseline_score + improvement_threshold)
        
        return improved, baseline_score, refined_score
    
    def _calculate_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate an overall score from metrics.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Overall score
        """
        # Use weights from metric configuration if available
        weights = self.metric_config.get('metric_weights', {})
        if not weights:
            # Default to equal weighting of all metrics
            metrics_list = list(metrics.keys())
            weights = {metric: 1.0 / len(metrics_list) for metric in metrics_list}
        
        # Calculate weighted sum
        score = 0.0
        for metric, value in metrics.items():
            if metric in weights:
                score += value * weights.get(metric, 0.0)
        
        return score
    
    def _save_iteration(
        self, 
        iteration: int, 
        system_prompt: str, 
        output_prompt: str, 
        metrics: Dict[str, Any],
        training_accuracy: Optional[float] = None,
        validation_accuracy: Optional[float] = None
    ) -> None:
        """Save iteration data to the database.
        
        Args:
            iteration: Iteration number
            system_prompt: System prompt for this iteration
            output_prompt: Output prompt for this iteration
            metrics: Metrics for this iteration
            training_accuracy: Optional training accuracy
            validation_accuracy: Optional validation accuracy
        """
        self.experiment_service.add_experiment_iteration(
            self.experiment_id,
            iteration,
            system_prompt,
            output_prompt,
            metrics,
            training_accuracy,
            validation_accuracy
        )