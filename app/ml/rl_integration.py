"""
Reinforcement Learning integration for ML prompt optimization in ATLAS.

This module provides functionality for:
1. Training RL agents to optimize prompts
2. Defining custom environments for prompt optimization
3. Evaluating prompts using RL-based approaches
"""

import gymnasium as gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import os
import json
import time
import logging
from datetime import datetime
import uuid
import pickle

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from app.ml.services import RLModelService, MLExperimentService

logger = logging.getLogger(__name__)

class PromptEngineeringEnv(gym.Env):
    """Custom Environment for prompt engineering using reinforcement learning."""
    
    def __init__(
        self,
        examples: List[Dict[str, Any]],
        base_system_prompt: str,
        base_output_prompt: str,
        llm_client=None,
        evaluator_client=None
    ):
        """Initialize the environment.
        
        Args:
            examples: List of examples for training/evaluation
            base_system_prompt: Initial system prompt
            base_output_prompt: Initial output prompt
            llm_client: Client for LLM API calls
            evaluator_client: Client for evaluation API calls
        """
        super(PromptEngineeringEnv, self).__init__()
        
        self.examples = examples
        self.base_system_prompt = base_system_prompt
        self.base_output_prompt = base_output_prompt
        self.llm_client = llm_client
        self.evaluator_client = evaluator_client
        
        # Track the current state
        self.current_system_prompt = base_system_prompt
        self.current_output_prompt = base_output_prompt
        self.current_score = 0.0
        self.step_count = 0
        self.max_steps = 20
        
        # Define action and observation spaces
        # Actions represent different prompt modification strategies
        # 0: Add reasoning steps
        # 1: Add more examples
        # 2: Make prompt more specific
        # 3: Simplify prompt
        # 4: Add step-by-step instructions
        self.action_space = spaces.Discrete(5)
        
        # Observation space: features of the current prompt and performance
        # [prompt_length, has_examples, has_reasoning, has_instructions, 
        #  current_score, steps_remaining]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([1000, 1, 1, 1, 1, self.max_steps]),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset to initial state
        self.current_system_prompt = self.base_system_prompt
        self.current_output_prompt = self.base_output_prompt
        self.step_count = 0
        
        # Evaluate the initial prompt
        self.current_score = self._evaluate_prompt(
            self.current_system_prompt, 
            self.current_output_prompt
        )
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment by modifying the prompts.
        
        Args:
            action: Integer action to take
            
        Returns:
            observation: New observation
            reward: Reward from the action
            terminated: Whether the episode is done
            truncated: Whether the episode is truncated
            info: Additional information
        """
        self.step_count += 1
        
        # Apply the selected action to modify the prompts
        new_system_prompt, new_output_prompt = self._apply_action(
            action, 
            self.current_system_prompt,
            self.current_output_prompt
        )
        
        # Evaluate the new prompts
        new_score = self._evaluate_prompt(new_system_prompt, new_output_prompt)
        
        # Calculate reward: improvement in score
        reward = new_score - self.current_score
        
        # Update the current state
        self.current_system_prompt = new_system_prompt
        self.current_output_prompt = new_output_prompt
        self.current_score = new_score
        
        # Check if we've reached the maximum steps
        done = self.step_count >= self.max_steps
        
        return self._get_observation(), reward, done, False, {
            'system_prompt': new_system_prompt,
            'output_prompt': new_output_prompt,
            'score': new_score
        }
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation.
        
        Returns:
            Numpy array representing the observation
        """
        # Extract prompt features
        prompt_length = min(len(self.current_system_prompt) + len(self.current_output_prompt), 1000)
        has_examples = 1.0 if "example" in self.current_system_prompt.lower() else 0.0
        has_reasoning = 1.0 if "reason" in self.current_system_prompt.lower() else 0.0
        has_instructions = 1.0 if "instruct" in self.current_system_prompt.lower() else 0.0
        
        # Normalize prompt length
        norm_prompt_length = prompt_length / 1000.0
        
        # Create the observation vector
        observation = np.array([
            norm_prompt_length,
            has_examples,
            has_reasoning,
            has_instructions,
            self.current_score,
            (self.max_steps - self.step_count) / self.max_steps
        ], dtype=np.float32)
        
        return observation
    
    def _apply_action(self, action: int, system_prompt: str, output_prompt: str) -> Tuple[str, str]:
        """Apply an action to modify the prompts.
        
        Args:
            action: The action to apply
            system_prompt: Current system prompt
            output_prompt: Current output prompt
            
        Returns:
            Tuple of (new_system_prompt, new_output_prompt)
        """
        # In a real implementation, we would use the LLM to generate modifications
        # For this example, we'll use simple templates
        
        if action == 0:  # Add reasoning steps
            if "reason step by step" not in system_prompt.lower():
                system_prompt += "\nMake sure to reason step by step before providing your answer."
                
        elif action == 1:  # Add more examples
            if "example" not in system_prompt.lower():
                system_prompt += "\nHere's an example of what I'm looking for: [EXAMPLE]"
                
        elif action == 2:  # Make prompt more specific
            if "specific" not in system_prompt.lower():
                system_prompt += "\nBe very specific and detailed in your response."
                
        elif action == 3:  # Simplify prompt
            # Mock simplification - in a real scenario, we would use an LLM to generate a simplified prompt
            if len(system_prompt) > 100:
                system_prompt = system_prompt[:int(len(system_prompt) * 0.8)]
                system_prompt += "\nKeep your response concise and to the point."
                
        elif action == 4:  # Add step-by-step instructions
            if "step-by-step" not in system_prompt.lower():
                system_prompt += "\nPlease provide your answer in a step-by-step format."
        
        return system_prompt, output_prompt
    
    def _evaluate_prompt(self, system_prompt: str, output_prompt: str) -> float:
        """Evaluate a prompt pair by generating responses and computing metrics.
        
        Args:
            system_prompt: System prompt to evaluate
            output_prompt: Output prompt to evaluate
            
        Returns:
            Score between 0 and 1
        """
        # In a real implementation, we would:
        # 1. Generate responses using the LLM
        # 2. Evaluate against ground truth
        # 3. Compute and return metrics
        
        # For this example, we'll return a score based on prompt features
        # This is just for demonstration - a real implementation would use actual LLM outputs
        
        combined = system_prompt + " " + output_prompt
        lower_combined = combined.lower()
        
        score = 0.5  # Base score
        
        # Add points for desired features
        if "reason" in lower_combined:
            score += 0.1
        if "example" in lower_combined:
            score += 0.1
        if "step" in lower_combined:
            score += 0.1
        if "instructions" in lower_combined or "instruct" in lower_combined:
            score += 0.1
        
        # Penalty for very long or very short prompts
        total_length = len(system_prompt) + len(output_prompt)
        if total_length < 50:
            score -= 0.1
        elif total_length > 500:
            score -= 0.1
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))


class RLPromptOptimizer:
    """Reinforcement Learning-based optimizer for prompt engineering."""
    
    def __init__(
        self, 
        model_id: str = None,
        model_type: str = "ppo",
        hyperparameters: Dict[str, Any] = None
    ):
        """Initialize the RL optimizer.
        
        Args:
            model_id: ID of an existing model to load, if provided
            model_type: Type of RL algorithm to use (ppo, a2c, dqn)
            hyperparameters: Dictionary of hyperparameters to use
        """
        self.model_id = model_id
        self.model = None
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0,
            'verbose': 0
        }
        
        # If model_id is provided, try to load the model
        if model_id:
            self._load_model()
    
    def _load_model(self) -> bool:
        """Load a saved model from the database.
        
        Returns:
            True if the model was loaded successfully, False otherwise
        """
        service = RLModelService()
        model_data = service.get_rl_model(self.model_id)
        
        if not model_data:
            logger.warning(f"Model with ID {self.model_id} not found")
            return False
        
        model_path = model_data.get('model_path')
        if not model_path or not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return False
        
        try:
            # Load the model based on type
            if self.model_type == "ppo":
                self.model = PPO.load(model_path)
            elif self.model_type == "a2c":
                self.model = A2C.load(model_path)
            elif self.model_type == "dqn":
                self.model = DQN.load(model_path)
            else:
                logger.warning(f"Unsupported model type: {self.model_type}")
                return False
            
            logger.info(f"Successfully loaded model from {model_path}")
            return True
        
        except Exception as e:
            logger.exception(f"Error loading model: {str(e)}")
            return False
    
    def train(
        self, 
        examples: List[Dict[str, Any]],
        base_system_prompt: str,
        base_output_prompt: str,
        total_timesteps: int = 10000,
        llm_client=None,
        evaluator_client=None
    ) -> Dict[str, Any]:
        """Train the RL model for prompt optimization.
        
        Args:
            examples: List of examples for training
            base_system_prompt: Initial system prompt
            base_output_prompt: Initial output prompt
            total_timesteps: Total number of training timesteps
            llm_client: Client for LLM API calls
            evaluator_client: Client for evaluation API calls
            
        Returns:
            Dictionary of training results
        """
        # Create the environment
        env = PromptEngineeringEnv(
            examples=examples,
            base_system_prompt=base_system_prompt,
            base_output_prompt=base_output_prompt,
            llm_client=llm_client,
            evaluator_client=evaluator_client
        )
        
        # Wrap with monitor for logging
        log_dir = os.path.join('data', 'logs', f'rl_training_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        
        # Create the model based on type
        if self.model_type == "ppo":
            self.model = PPO("MlpPolicy", env, **self.hyperparameters)
        elif self.model_type == "a2c":
            self.model = A2C("MlpPolicy", env, **self.hyperparameters)
        elif self.model_type == "dqn":
            self.model = DQN("MlpPolicy", env, **self.hyperparameters)
        else:
            return {'error': f"Unsupported model type: {self.model_type}"}
        
        # Train the model
        self.model.learn(total_timesteps=total_timesteps)
        
        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(self.model, env, n_eval_episodes=10)
        
        # Save the model
        model_path = os.path.join('data', 'models', f'rl_model_{uuid.uuid4()}')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        
        # Create or update the model in database
        service = RLModelService()
        metrics = {
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'total_timesteps': total_timesteps
        }
        
        if self.model_id:
            # Update existing model
            model_data = {
                'model_path': model_path,
                'metrics': metrics,
                'hyperparameters': self.hyperparameters,
                'updated_at': datetime.utcnow().isoformat()
            }
            service.update_rl_model(self.model_id, model_data)
        else:
            # Create new model
            model_data = {
                'name': f"{self.model_type.upper()} Model {datetime.utcnow().strftime('%Y-%m-%d')}",
                'model_type': self.model_type,
                'model_path': model_path,
                'metrics': metrics,
                'hyperparameters': self.hyperparameters,
                'action_space': {'n': 5},
                'observation_space': {'shape': [6], 'low': [0, 0, 0, 0, 0, 0], 'high': [1, 1, 1, 1, 1, 1]},
                'is_active': True
            }
            new_model = service.create_rl_model(model_data)
            self.model_id = new_model.get('id')
        
        return {
            'model_id': self.model_id,
            'metrics': metrics,
            'model_path': model_path
        }
    
    def optimize_prompt(
        self,
        system_prompt: str,
        output_prompt: str,
        examples: List[Dict[str, Any]],
        max_iterations: int = 5,
        llm_client=None,
        evaluator_client=None
    ) -> Dict[str, Any]:
        """Optimize a prompt using the trained RL model.
        
        Args:
            system_prompt: Initial system prompt
            output_prompt: Initial output prompt
            examples: List of examples for evaluation
            max_iterations: Maximum number of optimization iterations
            llm_client: Client for LLM API calls
            evaluator_client: Client for evaluation API calls
            
        Returns:
            Dictionary with optimization results
        """
        if not self.model:
            return {'error': 'Model not loaded'}
        
        # Create the environment
        env = PromptEngineeringEnv(
            examples=examples,
            base_system_prompt=system_prompt,
            base_output_prompt=output_prompt,
            llm_client=llm_client,
            evaluator_client=evaluator_client
        )
        
        # Run the optimization
        obs, _ = env.reset()
        
        system_prompts = [system_prompt]
        output_prompts = [output_prompt]
        scores = [env.current_score]
        
        done = False
        steps = 0
        
        while not done and steps < max_iterations:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Take the action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            # Store results
            system_prompts.append(info['system_prompt'])
            output_prompts.append(info['output_prompt'])
            scores.append(info['score'])
        
        # Find the best prompt
        best_idx = np.argmax(scores)
        
        return {
            'original_system_prompt': system_prompt,
            'original_output_prompt': output_prompt,
            'optimized_system_prompt': system_prompts[best_idx],
            'optimized_output_prompt': output_prompts[best_idx],
            'original_score': scores[0],
            'optimized_score': scores[best_idx],
            'improvement': scores[best_idx] - scores[0],
            'steps_taken': steps,
            'all_scores': scores
        }