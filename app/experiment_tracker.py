import os
import json
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """
    Tracks and saves experiment data, including prompts, metrics, and results.
    """
    
    def __init__(self, base_dir='experiments'):
        self.base_dir = base_dir
        self.metrics_dir = os.path.join(base_dir, 'metrics')
        self.prompt_history_dir = os.path.join(base_dir, 'prompts')
        
        # Create directories if they don't exist
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.prompt_history_dir, exist_ok=True)
    
    def start_experiment(self):
        """Start a new experiment and return the experiment ID."""
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return experiment_id
    
    def save_iteration(self, experiment_id, iteration, system_prompt, output_prompt, 
                       metrics, examples=None, optimizer_reasoning=None):
        """
        Save data for a training iteration.
        
        Args:
            experiment_id (str): Experiment identifier
            iteration (int): Iteration number
            system_prompt (str): System prompt used
            output_prompt (str): Output prompt used
            metrics (dict): Metrics from the evaluation
            examples (list, optional): Example results
            optimizer_reasoning (str, optional): Reasoning from the optimizer
        """
        try:
            # Create experiment directories if needed
            exp_dir = os.path.join(self.base_dir, experiment_id)
            os.makedirs(exp_dir, exist_ok=True)
            
            # Save prompts
            prompt_data = {
                "iteration": iteration,
                "timestamp": time.time(),
                "system_prompt": system_prompt,
                "output_prompt": output_prompt
            }
            prompts_file = os.path.join(exp_dir, f"prompts_{iteration}.json")
            with open(prompts_file, 'w') as f:
                json.dump(prompt_data, f, indent=2)
            
            # Save metrics
            metrics_data = {
                "iteration": iteration,
                "timestamp": time.time(),
                "metrics": metrics
            }
            metrics_file = os.path.join(exp_dir, f"metrics_{iteration}.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            # Save example results (if provided)
            if examples:
                examples_file = os.path.join(exp_dir, f"examples_{iteration}.json")
                with open(examples_file, 'w') as f:
                    json.dump(examples, f, indent=2)
            
            # Save optimizer reasoning (if provided)
            if optimizer_reasoning:
                reasoning_file = os.path.join(exp_dir, f"reasoning_{iteration}.txt")
                with open(reasoning_file, 'w') as f:
                    f.write(optimizer_reasoning)
            
            logger.info(f"Saved data for experiment {experiment_id}, iteration {iteration}")
            return True
        except Exception as e:
            logger.error(f"Error saving experiment data: {e}")
            return False
    
    def load_experiment_history(self, experiment_id=None):
        """
        Load experiment history.
        
        Args:
            experiment_id (str, optional): Specific experiment to load
            
        Returns:
            list: List of experiment data dictionaries
        """
        try:
            if experiment_id:
                # Load specific experiment
                exp_dir = os.path.join(self.base_dir, experiment_id)
                return self._load_experiment_data(exp_dir, experiment_id)
            else:
                # Load all experiments
                experiments = []
                for exp_id in os.listdir(self.base_dir):
                    exp_dir = os.path.join(self.base_dir, exp_id)
                    if os.path.isdir(exp_dir):
                        exp_data = self._load_experiment_data(exp_dir, exp_id)
                        if exp_data:
                            experiments.append(exp_data)
                return sorted(experiments, key=lambda x: x.get('timestamp', 0), reverse=True)
        except Exception as e:
            logger.error(f"Error loading experiment history: {e}")
            return []
    
    def _load_experiment_data(self, exp_dir, exp_id):
        """Helper to load data for a single experiment."""
        try:
            # Find the metrics files
            metrics_files = [f for f in os.listdir(exp_dir) if f.startswith('metrics_') and f.endswith('.json')]
            
            if not metrics_files:
                return None
            
            # Sort by iteration
            metrics_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            
            # Load the latest metrics
            latest_metrics_file = os.path.join(exp_dir, metrics_files[-1])
            with open(latest_metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            # Find the corresponding prompts file
            iteration = metrics_data.get('iteration', 0)
            prompts_file = os.path.join(exp_dir, f"prompts_{iteration}.json")
            
            if os.path.exists(prompts_file):
                with open(prompts_file, 'r') as f:
                    prompts_data = json.load(f)
            else:
                prompts_data = {}
            
            # Combine the data
            experiment_data = {
                "experiment_id": exp_id,
                "iteration": iteration,
                "timestamp": metrics_data.get('timestamp', 0),
                "metrics": metrics_data.get('metrics', {}),
                "system_prompt": prompts_data.get('system_prompt', ''),
                "output_prompt": prompts_data.get('output_prompt', '')
            }
            
            # Try to load optimizer reasoning if it exists
            reasoning_file = os.path.join(exp_dir, f"reasoning_{iteration}.txt")
            if os.path.exists(reasoning_file):
                with open(reasoning_file, 'r') as f:
                    experiment_data["reasoning"] = f.read()
            
            return experiment_data
        except Exception as e:
            logger.error(f"Error loading experiment {exp_id}: {e}")
            return None
    
    def get_iterations(self, experiment_id):
        """
        Get all iterations for a specific experiment.
        
        Args:
            experiment_id (str): Experiment identifier
            
        Returns:
            list: List of iteration data dictionaries
        """
        try:
            exp_dir = os.path.join(self.base_dir, experiment_id)
            if not os.path.exists(exp_dir):
                return []
            
            # Find all metrics files for this experiment
            metrics_files = [f for f in os.listdir(exp_dir) if f.startswith('metrics_') and f.endswith('.json')]
            
            iterations = []
            for metrics_file in metrics_files:
                iteration = int(metrics_file.split('_')[1].split('.')[0])
                
                # Load metrics
                with open(os.path.join(exp_dir, metrics_file), 'r') as f:
                    metrics_data = json.load(f)
                
                # Load prompts
                prompts_file = os.path.join(exp_dir, f"prompts_{iteration}.json")
                if os.path.exists(prompts_file):
                    with open(prompts_file, 'r') as f:
                        prompts_data = json.load(f)
                else:
                    prompts_data = {}
                
                # Load reasoning if it exists
                reasoning = ""
                reasoning_file = os.path.join(exp_dir, f"reasoning_{iteration}.txt")
                if os.path.exists(reasoning_file):
                    with open(reasoning_file, 'r') as f:
                        reasoning = f.read()
                
                iterations.append({
                    "iteration": iteration,
                    "timestamp": metrics_data.get('timestamp', 0),
                    "metrics": metrics_data.get('metrics', {}),
                    "system_prompt": prompts_data.get('system_prompt', ''),
                    "output_prompt": prompts_data.get('output_prompt', ''),
                    "reasoning": reasoning,
                    "optimizer_reasoning": reasoning  # Add this for backward compatibility with tests
                })
            
            # Sort by iteration
            return sorted(iterations, key=lambda x: x['iteration'])
        except Exception as e:
            logger.error(f"Error getting iterations for experiment {experiment_id}: {e}")
            return []