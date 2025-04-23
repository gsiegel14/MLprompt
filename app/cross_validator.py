
"""
Cross-Validation module for prompt optimization.

This module implements k-fold cross-validation to get more reliable 
estimates of prompt performance across different data distributions.
"""

import os
import json
import logging
import random
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class CrossValidator:
    """
    K-fold cross-validation for prompt optimization.
    """
    
    def __init__(self, data_module, workflow, folds=5):
        """
        Initialize the cross-validator.
        
        Args:
            data_module: DataModule instance
            workflow: PromptOptimizationWorkflow instance
            folds: Number of folds for cross-validation
        """
        self.data_module = data_module
        self.workflow = workflow
        self.folds = folds
        
        # Create directory for validation results
        self.results_dir = 'cross_validation'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def split_data_into_folds(self, examples: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Split examples into k equal-sized folds.
        
        Args:
            examples: List of examples to split
            
        Returns:
            List of example lists, one per fold
        """
        # Shuffle the examples
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        # Calculate fold size
        fold_size = len(shuffled) // self.folds
        
        # Split into folds
        folds = []
        for i in range(self.folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < self.folds - 1 else len(shuffled)
            folds.append(shuffled[start_idx:end_idx])
        
        return folds
    
    def evaluate_system(self, system_prompt: str, output_prompt: str, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a prompt system using k-fold cross-validation.
        
        Args:
            system_prompt: System prompt to evaluate
            output_prompt: Output prompt to evaluate
            examples: List of examples to use
            
        Returns:
            Dict containing cross-validation metrics
        """
        logger.info(f"Starting {self.folds}-fold cross-validation")
        
        # Split data into folds
        folds = self.split_data_into_folds(examples)
        logger.info(f"Split data into {len(folds)} folds with ~{len(folds[0])} examples per fold")
        
        # Run validation for each fold
        fold_metrics = []
        
        for i in range(self.folds):
            logger.info(f"Validating fold {i+1}/{self.folds}")
            
            # Create validation set (current fold)
            validation_set = folds[i]
            
            # Create training set (all other folds)
            training_set = []
            for j in range(self.folds):
                if j != i:
                    training_set.extend(folds[j])
            
            # Temporarily override the data module's examples
            orig_train = self.data_module.get_train_examples()
            orig_validation = self.data_module.get_validation_examples()
            
            self.data_module._train_examples = training_set
            self.data_module._validation_examples = validation_set
            
            try:
                # Run the workflow with just 1 iteration to evaluate current prompts
                result = self.workflow.run_training_cycle(
                    system_prompt=system_prompt,
                    output_prompt=output_prompt,
                    max_iterations=1,
                    batch_size=min(10, len(validation_set))
                )
                
                # Extract metrics
                if 'best_score' in result:
                    fold_metrics.append({
                        'fold': i+1,
                        'score': result['best_score'],
                        'validation_size': len(validation_set),
                        'training_size': len(training_set)
                    })
            except Exception as e:
                logger.error(f"Error validating fold {i+1}: {e}")
            finally:
                # Restore original examples
                self.data_module._train_examples = orig_train
                self.data_module._validation_examples = orig_validation
        
        # Calculate aggregate metrics
        if not fold_metrics:
            logger.error("No valid metrics from any fold")
            return {"error": "Cross-validation failed to produce valid metrics"}
        
        avg_score = sum(m['score'] for m in fold_metrics) / len(fold_metrics)
        scores = [m['score'] for m in fold_metrics]
        
        validation_results = {
            "average_score": avg_score,
            "fold_metrics": fold_metrics,
            "min_score": min(scores),
            "max_score": max(scores),
            "score_variance": sum((s - avg_score) ** 2 for s in scores) / len(scores),
            "folds": self.folds,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"cross_validation_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Cross-validation complete. Average score: {avg_score:.4f}")
        logger.info(f"Results saved to {results_file}")
        
        return validation_results
    
    def compare_systems(self, systems: List[Dict[str, str]], examples: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Compare multiple prompt systems using cross-validation.
        
        Args:
            systems: List of dicts with 'system_prompt', 'output_prompt', and 'name' keys
            examples: Optional list of examples (uses data_module if None)
            
        Returns:
            Dict with comparison results
        """
        if examples is None:
            # Get all examples from data module
            train_examples = self.data_module.get_train_examples() or []
            validation_examples = self.data_module.get_validation_examples() or []
            examples = train_examples + validation_examples
        
        if not examples:
            logger.error("No examples available for cross-validation")
            return {"error": "No examples available"}
        
        # Evaluate each system
        system_results = []
        
        for i, system in enumerate(systems):
            logger.info(f"Evaluating system {i+1}/{len(systems)}: {system.get('name', f'System {i+1}')}")
            
            try:
                system_prompt = system.get('system_prompt', '')
                output_prompt = system.get('output_prompt', '')
                
                if not system_prompt or not output_prompt:
                    logger.warning(f"Skipping system {i+1} due to missing prompts")
                    continue
                
                results = self.evaluate_system(system_prompt, output_prompt, examples)
                
                system_results.append({
                    "name": system.get('name', f'System {i+1}'),
                    "results": results
                })
            except Exception as e:
                logger.error(f"Error evaluating system {i+1}: {e}")
        
        # Create comparison report
        comparison = {
            "systems": system_results,
            "timestamp": datetime.now().isoformat(),
            "example_count": len(examples)
        }
        
        # Find the best system
        if system_results:
            best_system_idx = max(range(len(system_results)), 
                                key=lambda i: system_results[i]['results'].get('average_score', 0))
            best_system = system_results[best_system_idx]
            
            comparison["best_system"] = {
                "name": best_system['name'],
                "average_score": best_system['results'].get('average_score', 0),
                "index": best_system_idx
            }
        
        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = os.path.join(self.results_dir, f"comparison_{timestamp}.json")
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"System comparison complete. Results saved to {comparison_file}")
        
        return comparison
