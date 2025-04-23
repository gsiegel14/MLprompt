
"""
Hyperparameter tuning module for the prompt optimization workflow.

This module provides tools for systematically exploring different 
hyperparameter configurations to optimize the prompt refinement process.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import itertools

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """
    Manages systematic tuning of hyperparameters for prompt optimization.
    """
    
    def __init__(self, base_dir='hyperparameter_tuning'):
        """Initialize the tuner."""
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Default hyperparameter search space
        self.default_search_space = {
            'temperature': [0.0, 0.3, 0.7],
            'optimizer_strategy': ['reasoning_first', 'full_rewrite', 'targeted_edit'],
            'batch_size': [3, 5, 10],
            'max_iterations': [1, 2, 3]
        }
    
    def generate_configurations(self, search_space: Optional[Dict[str, List]] = None) -> List[Dict[str, Any]]:
        """
        Generate all possible hyperparameter configurations from the search space.
        
        Args:
            search_space: Dictionary mapping parameter names to lists of possible values
            
        Returns:
            List of configuration dictionaries
        """
        if search_space is None:
            search_space = self.default_search_space
        
        # Get parameter names and possible values
        param_names = list(search_space.keys())
        param_values = [search_space[name] for name in param_names]
        
        # Generate all combinations
        configurations = []
        for combination in itertools.product(*param_values):
            config = {name: value for name, value in zip(param_names, combination)}
            configurations.append(config)
        
        logger.info(f"Generated {len(configurations)} hyperparameter configurations")
        return configurations
    
    def save_results(self, experiment_id: str, config: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """
        Save the results of a hyperparameter configuration run.
        
        Args:
            experiment_id: ID of the experiment
            config: Hyperparameter configuration used
            metrics: Resulting metrics
        """
        # Create experiment directory
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(self.base_dir, experiment_id)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save configuration and results
        result_data = {
            "configuration": config,
            "metrics": metrics,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat()
        }
        
        result_file = os.path.join(experiment_dir, f"results_{run_id}.json")
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Saved hyperparameter tuning results to {result_file}")
    
    def find_best_configuration(self, experiment_id: str, metric_key: str = 'avg_score') -> Dict[str, Any]:
        """
        Find the best hyperparameter configuration based on a specific metric.
        
        Args:
            experiment_id: ID of the experiment
            metric_key: Key of the metric to optimize
            
        Returns:
            Best configuration with its metrics
        """
        experiment_dir = os.path.join(self.base_dir, experiment_id)
        if not os.path.exists(experiment_dir):
            logger.warning(f"No results found for experiment {experiment_id}")
            return {}
        
        # Load all result files
        result_files = [f for f in os.listdir(experiment_dir) if f.startswith('results_') and f.endswith('.json')]
        
        if not result_files:
            logger.warning(f"No result files found in {experiment_dir}")
            return {}
        
        # Track the best configuration
        best_score = 0
        best_result = None
        
        for result_file in result_files:
            with open(os.path.join(experiment_dir, result_file), 'r') as f:
                result = json.load(f)
            
            # Extract the metric to optimize
            score = result.get('metrics', {}).get(metric_key, 0)
            
            if score > best_score:
                best_score = score
                best_result = result
        
        if best_result:
            logger.info(f"Found best configuration with {metric_key} = {best_score}")
            return best_result
        else:
            logger.warning(f"No valid results found for metric {metric_key}")
            return {}
    
    def run_grid_search(self, workflow, system_prompt: str, output_prompt: str, 
                       search_space: Optional[Dict[str, List]] = None,
                       metric_key: str = 'avg_score') -> Dict[str, Any]:
        """
        Run a grid search over hyperparameter configurations.
        
        Args:
            workflow: The workflow instance to use
            system_prompt: The system prompt to use
            output_prompt: The output prompt to use
            search_space: Dictionary mapping parameter names to lists of possible values
            metric_key: Key of the metric to optimize
            
        Returns:
            Best configuration with its metrics
        """
        # Generate configurations
        configurations = self.generate_configurations(search_space)
        
        # Create an experiment ID for this tuning run
        experiment_id = f"hypertuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create a summary file
        summary_file = os.path.join(self.base_dir, f"{experiment_id}_summary.json")
        
        # Track the best configuration
        best_score = 0
        best_config = None
        
        # Run each configuration
        for i, config in enumerate(configurations):
            logger.info(f"Running configuration {i+1}/{len(configurations)}: {config}")
            
            try:
                # Run the workflow with this configuration
                result = workflow.run_training_cycle(
                    system_prompt=system_prompt,
                    output_prompt=output_prompt,
                    max_iterations=config.get('max_iterations', 1),
                    batch_size=config.get('batch_size', 5),
                    optimizer_strategy=config.get('optimizer_strategy', 'reasoning_first'),
                    optimizer_type=config.get('optimizer_type', 'general')
                )
                
                # Extract metrics
                if 'best_score' in result:
                    metrics = {'best_score': result['best_score']}
                    
                    # Save result
                    self.save_results(experiment_id, config, metrics)
                    
                    # Update best configuration
                    if metrics.get('best_score', 0) > best_score:
                        best_score = metrics['best_score']
                        best_config = config
                
            except Exception as e:
                logger.error(f"Error running configuration {config}: {e}")
        
        # Save summary
        summary = {
            "experiment_id": experiment_id,
            "search_space": search_space or self.default_search_space,
            "best_configuration": best_config,
            "best_score": best_score,
            "total_configurations": len(configurations),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Completed hyperparameter grid search. Best score: {best_score}")
        return summary
