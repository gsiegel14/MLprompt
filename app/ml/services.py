"""
Service layer for ML-related operations in the ATLAS platform.

This module contains services for managing ML configurations, 
running experiments, and handling model operations.
"""

from app import db
from app.ml.models import (
    ModelConfiguration, 
    MetricConfiguration, 
    MLExperiment, 
    MLExperimentIteration,
    MetaLearningModel,
    RLModel
)
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

class MLSettingsService:
    """Service for managing ML configurations and settings."""

    @staticmethod
    def get_model_configurations(user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get model configurations, optionally filtered by user.
        
        Args:
            user_id: Optional user ID to filter configurations by
            
        Returns:
            List of model configurations as dictionaries
        """
        query = ModelConfiguration.query
        if user_id:
            query = query.filter(ModelConfiguration.user_id == user_id)
        
        configurations = query.all()
        return [config.to_dict() for config in configurations]
    
    @staticmethod
    def get_model_configuration(config_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific model configuration by ID.
        
        Args:
            config_id: The ID of the model configuration
            
        Returns:
            Model configuration as a dictionary, or None if not found
        """
        config = ModelConfiguration.query.filter_by(id=config_id).first()
        return config.to_dict() if config else None
    
    @staticmethod
    def create_model_configuration(data: Dict[str, Any], user_id: Optional[int] = None) -> Dict[str, Any]:
        """Create a new model configuration.
        
        Args:
            data: Dictionary containing model configuration data
            user_id: Optional user ID to associate with the configuration
            
        Returns:
            The created model configuration as a dictionary
        """
        # If this is set as default, unset any existing defaults
        if data.get('is_default', False):
            query = ModelConfiguration.query
            if user_id:
                query = query.filter(ModelConfiguration.user_id == user_id)
            
            default_configs = query.filter_by(is_default=True).all()
            for config in default_configs:
                config.is_default = False
        
        # Create the new configuration
        config = ModelConfiguration(
            id=str(uuid.uuid4()),
            name=data.get('name', 'New Configuration'),
            primary_model=data.get('primary_model', 'gemini-1.5-flash'),
            optimizer_model=data.get('optimizer_model', 'gemini-1.5-flash'),
            temperature=data.get('temperature', 0.0),
            max_tokens=data.get('max_tokens', 1024),
            top_p=data.get('top_p', 1.0),
            top_k=data.get('top_k', 40),
            is_default=data.get('is_default', False),
            user_id=user_id
        )
        
        db.session.add(config)
        db.session.commit()
        
        return config.to_dict()
    
    @staticmethod
    def update_model_configuration(config_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing model configuration.
        
        Args:
            config_id: The ID of the model configuration to update
            data: Dictionary containing updated configuration data
            
        Returns:
            The updated model configuration as a dictionary, or None if not found
        """
        config = ModelConfiguration.query.filter_by(id=config_id).first()
        if not config:
            return None
        
        # If this is being set as default, unset any existing defaults
        if data.get('is_default', False) and not config.is_default:
            query = ModelConfiguration.query
            if config.user_id:
                query = query.filter(ModelConfiguration.user_id == config.user_id)
            
            default_configs = query.filter_by(is_default=True).all()
            for default_config in default_configs:
                default_config.is_default = False
        
        # Update the configuration
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        db.session.commit()
        return config.to_dict()
    
    @staticmethod
    def delete_model_configuration(config_id: str) -> bool:
        """Delete a model configuration.
        
        Args:
            config_id: The ID of the model configuration to delete
            
        Returns:
            True if the configuration was deleted, False otherwise
        """
        config = ModelConfiguration.query.filter_by(id=config_id).first()
        if not config:
            return False
        
        db.session.delete(config)
        db.session.commit()
        return True
    
    @staticmethod
    def get_metric_configurations(user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get metric configurations, optionally filtered by user.
        
        Args:
            user_id: Optional user ID to filter configurations by
            
        Returns:
            List of metric configurations as dictionaries
        """
        query = MetricConfiguration.query
        if user_id:
            query = query.filter(MetricConfiguration.user_id == user_id)
        
        configurations = query.all()
        return [config.to_dict() for config in configurations]
    
    @staticmethod
    def create_metric_configuration(data: Dict[str, Any], user_id: Optional[int] = None) -> Dict[str, Any]:
        """Create a new metric configuration.
        
        Args:
            data: Dictionary containing metric configuration data
            user_id: Optional user ID to associate with the configuration
            
        Returns:
            The created metric configuration as a dictionary
        """
        config = MetricConfiguration(
            id=str(uuid.uuid4()),
            name=data.get('name', 'New Metrics'),
            metrics=data.get('metrics', ['exact_match', 'semantic_similarity']),
            metric_weights=data.get('metric_weights', {}),
            target_threshold=data.get('target_threshold', 0.8),
            user_id=user_id
        )
        
        db.session.add(config)
        db.session.commit()
        
        return config.to_dict()

class MLExperimentService:
    """Service for managing ML experiments and workflows."""
    
    @staticmethod
    def get_experiments(user_id: Optional[int] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get ML experiments, optionally filtered by user.
        
        Args:
            user_id: Optional user ID to filter experiments by
            limit: Maximum number of experiments to return
            
        Returns:
            List of experiments as dictionaries
        """
        query = MLExperiment.query.order_by(MLExperiment.created_at.desc())
        if user_id:
            query = query.filter(MLExperiment.user_id == user_id)
        
        experiments = query.limit(limit).all()
        return [exp.to_dict() for exp in experiments]
    
    @staticmethod
    def get_experiment(experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific experiment by ID.
        
        Args:
            experiment_id: The ID of the experiment
            
        Returns:
            Experiment as a dictionary, or None if not found
        """
        experiment = MLExperiment.query.filter_by(id=experiment_id).first()
        return experiment.to_dict() if experiment else None
    
    @staticmethod
    def create_experiment(
        name: str, 
        description: Optional[str] = None,
        model_config_id: Optional[str] = None,
        metric_config_id: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a new ML experiment.
        
        Args:
            name: Name of the experiment
            description: Optional description of the experiment
            model_config_id: Optional ID of the model configuration to use
            metric_config_id: Optional ID of the metric configuration to use
            user_id: Optional user ID to associate with the experiment
            
        Returns:
            The created experiment as a dictionary
        """
        # If model_config_id is not provided, use the default configuration
        if not model_config_id:
            default_config = ModelConfiguration.query.filter_by(is_default=True).first()
            model_config_id = default_config.id if default_config else None
        
        # If metric_config_id is not provided, use the first available configuration
        if not metric_config_id:
            metric_config = MetricConfiguration.query.first()
            metric_config_id = metric_config.id if metric_config else None
        
        experiment = MLExperiment(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            model_config_id=model_config_id,
            metric_config_id=metric_config_id,
            status='created',
            user_id=user_id
        )
        
        db.session.add(experiment)
        db.session.commit()
        
        return experiment.to_dict()
    
    @staticmethod
    def update_experiment_status(experiment_id: str, status: str) -> Optional[Dict[str, Any]]:
        """Update the status of an experiment.
        
        Args:
            experiment_id: The ID of the experiment
            status: The new status (created, running, completed, failed)
            
        Returns:
            The updated experiment as a dictionary, or None if not found
        """
        experiment = MLExperiment.query.filter_by(id=experiment_id).first()
        if not experiment:
            return None
        
        experiment.status = status
        experiment.updated_at = datetime.utcnow()
        db.session.commit()
        
        return experiment.to_dict()
    
    @staticmethod
    def add_experiment_iteration(
        experiment_id: str,
        iteration_number: int,
        system_prompt: str,
        output_prompt: str,
        metrics: Dict[str, Any],
        training_accuracy: Optional[float] = None,
        validation_accuracy: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Add a new iteration to an experiment.
        
        Args:
            experiment_id: The ID of the experiment
            iteration_number: The iteration number
            system_prompt: The system prompt used in this iteration
            output_prompt: The output prompt used in this iteration
            metrics: Dictionary of metrics for this iteration
            training_accuracy: Optional training accuracy
            validation_accuracy: Optional validation accuracy
            
        Returns:
            The created iteration as a dictionary, or None if the experiment is not found
        """
        experiment = MLExperiment.query.filter_by(id=experiment_id).first()
        if not experiment:
            return None
        
        iteration = MLExperimentIteration(
            id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            iteration_number=iteration_number,
            system_prompt=system_prompt,
            output_prompt=output_prompt,
            metrics=metrics,
            training_accuracy=training_accuracy,
            validation_accuracy=validation_accuracy
        )
        
        db.session.add(iteration)
        
        # Update experiment status and result data
        experiment.status = 'running'
        if 'iterations' not in experiment.result_data:
            experiment.result_data['iterations'] = []
            
        experiment.result_data['iterations'].append({
            'iteration_number': iteration_number,
            'metrics': metrics,
            'training_accuracy': training_accuracy,
            'validation_accuracy': validation_accuracy
        })
        
        experiment.updated_at = datetime.utcnow()
        db.session.commit()
        
        return iteration.to_dict()
    
    @staticmethod
    def get_experiment_iterations(experiment_id: str) -> List[Dict[str, Any]]:
        """Get all iterations for an experiment.
        
        Args:
            experiment_id: The ID of the experiment
            
        Returns:
            List of iterations as dictionaries
        """
        iterations = MLExperimentIteration.query.filter_by(experiment_id=experiment_id).order_by(
            MLExperimentIteration.iteration_number).all()
        
        return [iteration.to_dict() for iteration in iterations]
    
    @staticmethod
    def complete_experiment(experiment_id: str, result_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Mark an experiment as completed and store its results.
        
        Args:
            experiment_id: The ID of the experiment
            result_data: Dictionary of result data
            
        Returns:
            The updated experiment as a dictionary, or None if not found
        """
        experiment = MLExperiment.query.filter_by(id=experiment_id).first()
        if not experiment:
            return None
        
        experiment.status = 'completed'
        experiment.result_data = result_data
        experiment.updated_at = datetime.utcnow()
        db.session.commit()
        
        return experiment.to_dict()

class MetaLearningService:
    """Service for managing meta-learning models and operations."""
    
    @staticmethod
    def get_meta_learning_models(user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get meta-learning models, optionally filtered by user.
        
        Args:
            user_id: Optional user ID to filter models by
            
        Returns:
            List of meta-learning models as dictionaries
        """
        query = MetaLearningModel.query
        if user_id:
            query = query.filter(MetaLearningModel.user_id == user_id)
        
        models = query.all()
        return [model.to_dict() for model in models]
    
    @staticmethod
    def create_meta_learning_model(data: Dict[str, Any], user_id: Optional[int] = None) -> Dict[str, Any]:
        """Create a new meta-learning model.
        
        Args:
            data: Dictionary containing meta-learning model data
            user_id: Optional user ID to associate with the model
            
        Returns:
            The created meta-learning model as a dictionary
        """
        model = MetaLearningModel(
            id=str(uuid.uuid4()),
            name=data.get('name', 'New Meta-Learning Model'),
            model_type=data.get('model_type', 'lightgbm'),
            hyperparameters=data.get('hyperparameters', {}),
            feature_config=data.get('feature_config', {}),
            is_active=data.get('is_active', False),
            user_id=user_id
        )
        
        db.session.add(model)
        db.session.commit()
        
        return model.to_dict()
    
    @staticmethod
    def update_meta_learning_model(model_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing meta-learning model.
        
        Args:
            model_id: The ID of the meta-learning model to update
            data: Dictionary containing updated model data
            
        Returns:
            The updated meta-learning model as a dictionary, or None if not found
        """
        model = MetaLearningModel.query.filter_by(id=model_id).first()
        if not model:
            return None
        
        for key, value in data.items():
            if hasattr(model, key):
                setattr(model, key, value)
        
        model.updated_at = datetime.utcnow()
        db.session.commit()
        
        return model.to_dict()

class RLModelService:
    """Service for managing reinforcement learning models and operations."""
    
    @staticmethod
    def get_rl_models(user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get RL models, optionally filtered by user.
        
        Args:
            user_id: Optional user ID to filter models by
            
        Returns:
            List of RL models as dictionaries
        """
        query = RLModel.query
        if user_id:
            query = query.filter(RLModel.user_id == user_id)
        
        models = query.all()
        return [model.to_dict() for model in models]
    
    @staticmethod
    def create_rl_model(data: Dict[str, Any], user_id: Optional[int] = None) -> Dict[str, Any]:
        """Create a new RL model.
        
        Args:
            data: Dictionary containing RL model data
            user_id: Optional user ID to associate with the model
            
        Returns:
            The created RL model as a dictionary
        """
        model = RLModel(
            id=str(uuid.uuid4()),
            name=data.get('name', 'New RL Model'),
            model_type=data.get('model_type', 'ppo'),
            hyperparameters=data.get('hyperparameters', {}),
            action_space=data.get('action_space', {}),
            observation_space=data.get('observation_space', {}),
            is_active=data.get('is_active', False),
            user_id=user_id
        )
        
        db.session.add(model)
        db.session.commit()
        
        return model.to_dict()