"""
LightGBM integration for ML prompt optimization in the ATLAS platform.

This module provides functionality for:
1. Training LightGBM models on prompt optimization data
2. Feature extraction from prompts and examples
3. Prediction of optimal prompt strategies
"""

import lightgbm as lgb
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

from app.ml.services import MLExperimentService, MetaLearningService

logger = logging.getLogger(__name__)

class LightGBMOptimizer:
    """LightGBM-based optimizer for prompt engineering."""
    
    def __init__(
        self, 
        model_id: str = None,
        hyperparameters: Dict[str, Any] = None
    ):
        """Initialize the LightGBM optimizer.
        
        Args:
            model_id: ID of an existing model to load, if provided
            hyperparameters: Dictionary of hyperparameters to use
        """
        self.model_id = model_id
        self.model = None
        self.feature_names = []
        self.hyperparameters = hyperparameters or {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # If model_id is provided, try to load the model
        if model_id:
            self._load_model()
    
    def _load_model(self) -> bool:
        """Load a saved model from the database.
        
        Returns:
            True if the model was loaded successfully, False otherwise
        """
        service = MetaLearningService()
        model_data = service.get_meta_learning_model(self.model_id)
        
        if not model_data:
            logger.warning(f"Model with ID {self.model_id} not found")
            return False
        
        model_path = model_data.get('model_path')
        if not model_path or not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return False
        
        try:
            # Load the model
            with open(model_path, 'rb') as f:
                model_dict = pickle.load(f)
                
            self.model = model_dict.get('model')
            self.feature_names = model_dict.get('feature_names', [])
            self.hyperparameters = model_dict.get('hyperparameters', self.hyperparameters)
            
            logger.info(f"Successfully loaded model from {model_path}")
            return True
        
        except Exception as e:
            logger.exception(f"Error loading model: {str(e)}")
            return False
    
    def extract_features(self, prompt: str, examples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract features from a prompt and examples.
        
        Args:
            prompt: The prompt text
            examples: List of examples
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Prompt-based features
        features['prompt_length'] = len(prompt)
        features['prompt_word_count'] = len(prompt.split())
        features['prompt_has_instructions'] = 1 if 'instruct' in prompt.lower() else 0
        features['prompt_has_examples'] = 1 if 'example' in prompt.lower() else 0
        features['prompt_has_reasoning'] = 1 if 'reason' in prompt.lower() else 0
        
        # Example-based features
        features['example_count'] = len(examples)
        
        if examples:
            input_lengths = [len(ex.get('user_input', '')) for ex in examples]
            output_lengths = [len(ex.get('ground_truth_output', '')) for ex in examples]
            
            features['avg_input_length'] = sum(input_lengths) / len(input_lengths)
            features['avg_output_length'] = sum(output_lengths) / len(output_lengths)
            features['max_input_length'] = max(input_lengths)
            features['min_input_length'] = min(input_lengths)
        else:
            features['avg_input_length'] = 0
            features['avg_output_length'] = 0
            features['max_input_length'] = 0
            features['min_input_length'] = 0
        
        return features
    
    def prepare_training_data(self, experiment_ids: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data from experiment results.
        
        Args:
            experiment_ids: List of experiment IDs to include
            
        Returns:
            Tuple of (X, y) where X is feature dataframe and y is target array
        """
        service = MLExperimentService()
        all_data = []
        all_labels = []
        
        for experiment_id in experiment_ids:
            experiment = service.get_experiment(experiment_id)
            if not experiment:
                logger.warning(f"Experiment with ID {experiment_id} not found")
                continue
            
            iterations = service.get_experiment_iterations(experiment_id)
            
            for i, iteration in enumerate(iterations):
                # Skip the first iteration (no improvement to measure yet)
                if i == 0:
                    continue
                
                # Extract data from the current iteration
                system_prompt = iteration.get('system_prompt', '')
                output_prompt = iteration.get('output_prompt', '')
                combined_prompt = f"{system_prompt}\n\n{output_prompt}"
                
                # Get the metrics
                metrics = iteration.get('metrics', {})
                
                # Determine if this iteration showed improvement
                improved = False
                if 'refined' in metrics and 'baseline' in metrics:
                    baseline_score = (
                        metrics['baseline'].get('exact_match', 0) + 
                        metrics['baseline'].get('semantic_similarity', 0)
                    ) / 2
                    
                    refined_score = (
                        metrics['refined'].get('exact_match', 0) + 
                        metrics['refined'].get('semantic_similarity', 0)
                    ) / 2
                    
                    improved = refined_score > baseline_score
                
                # Extract examples from the experiment
                examples = []  # In a real implementation, we'd extract actual examples
                
                # Extract features
                features = self.extract_features(combined_prompt, examples)
                
                # Add to our dataset
                all_data.append(features)
                all_labels.append(1 if improved else 0)
        
        # Convert to pandas dataframe and numpy array
        X = pd.DataFrame(all_data)
        y = np.array(all_labels)
        
        # Save feature names
        self.feature_names = list(X.columns)
        
        return X, y
    
    def train(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Train the LightGBM model on experiment results.
        
        Args:
            experiment_ids: List of experiment IDs to include in training
            
        Returns:
            Dictionary of training results
        """
        # Prepare the training data
        X, y = self.prepare_training_data(experiment_ids)
        
        if len(X) == 0:
            logger.warning("No training data available")
            return {'error': 'No training data available'}
        
        # Train test split
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=list(X.columns))
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train the model
        self.model = lgb.train(
            self.hyperparameters,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(10)]
        )
        
        # Evaluate the model
        y_pred = self.model.predict(X_val)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred_binary),
            'precision': precision_score(y_val, y_pred_binary, zero_division=0),
            'recall': recall_score(y_val, y_pred_binary, zero_division=0),
            'f1': f1_score(y_val, y_pred_binary, zero_division=0)
        }
        
        # Get feature importance
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = dict(zip(self.feature_names, importance.tolist()))
        
        # Save the model
        model_path = os.path.join('data', 'models', f'lgb_model_{uuid.uuid4()}.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Create a dictionary with model and metadata
        model_dict = {
            'model': self.model,
            'feature_names': self.feature_names,
            'hyperparameters': self.hyperparameters,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
        
        # Save to disk
        with open(model_path, 'wb') as f:
            pickle.dump(model_dict, f)
        
        # Create or update the model in database
        service = MetaLearningService()
        
        if self.model_id:
            # Update existing model
            model_data = {
                'model_path': model_path,
                'metrics': metrics,
                'feature_config': {'feature_names': self.feature_names},
                'hyperparameters': self.hyperparameters,
                'updated_at': datetime.utcnow().isoformat()
            }
            service.update_meta_learning_model(self.model_id, model_data)
        else:
            # Create new model
            model_data = {
                'name': f"LightGBM Model {datetime.utcnow().strftime('%Y-%m-%d')}",
                'model_type': 'lightgbm',
                'model_path': model_path,
                'metrics': metrics,
                'feature_config': {'feature_names': self.feature_names},
                'hyperparameters': self.hyperparameters,
                'is_active': True
            }
            new_model = service.create_meta_learning_model(model_data)
            self.model_id = new_model.get('id')
        
        return {
            'model_id': self.model_id,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'model_path': model_path
        }
    
    def predict(self, prompt: str, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict whether a prompt will lead to improvement.
        
        Args:
            prompt: The prompt text
            examples: List of examples
            
        Returns:
            Dictionary with prediction results
        """
        if not self.model:
            return {'error': 'Model not loaded'}
        
        # Extract features
        features = self.extract_features(prompt, examples)
        
        # Convert to dataframe with correct column order
        df = pd.DataFrame([features])
        df = df.reindex(columns=self.feature_names, fill_value=0)
        
        # Make prediction
        probability = self.model.predict(df)[0]
        prediction = 1 if probability > 0.5 else 0
        
        return {
            'probability': float(probability),
            'prediction': int(prediction),
            'will_improve': bool(prediction),
            'features_used': features
        }
    
    def get_improvement_suggestions(self, prompt: str, examples: List[Dict[str, Any]]) -> List[str]:
        """Get suggestions for improving a prompt.
        
        Args:
            prompt: The prompt text
            examples: List of examples
            
        Returns:
            List of improvement suggestions
        """
        if not self.model:
            return ['Model not loaded']
        
        # Extract features
        features = self.extract_features(prompt, examples)
        
        # Get feature importance
        importance = self.model.feature_importance(importance_type='gain')
        feature_imp_dict = dict(zip(self.feature_names, importance))
        
        # Sort features by importance
        sorted_features = sorted(feature_imp_dict.items(), key=lambda x: x[1], reverse=True)
        
        suggestions = []
        
        # Logic to generate suggestions based on important features
        for feature, importance in sorted_features[:5]:
            if feature == 'prompt_length' and features['prompt_length'] < 200:
                suggestions.append("Consider making your prompt longer with more details")
            
            elif feature == 'prompt_has_examples' and features['prompt_has_examples'] == 0:
                suggestions.append("Include examples in your prompt")
            
            elif feature == 'prompt_has_reasoning' and features['prompt_has_reasoning'] == 0:
                suggestions.append("Add reasoning steps to your prompt")
            
            elif feature == 'prompt_has_instructions' and features['prompt_has_instructions'] == 0:
                suggestions.append("Include clear instructions in your prompt")
            
            elif feature == 'example_count' and features['example_count'] < 3:
                suggestions.append("Add more examples to improve model understanding")
        
        return suggestions