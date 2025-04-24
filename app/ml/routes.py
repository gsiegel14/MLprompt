"""
API routes for ML-related operations in the ATLAS platform.

This module defines routes for:
- ML configurations management
- Experiment execution and monitoring
- 5-API workflow execution
- ML model training and management
"""

from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user
import json
import logging
from typing import Dict, Any, List, Optional

from app.ml.services import (
    MLSettingsService, 
    MLExperimentService, 
    MetaLearningService,
    RLModelService
)
from app.ml.workflow import FiveAPIWorkflow

logger = logging.getLogger(__name__)

ml_api = Blueprint('ml_api', __name__, url_prefix='/api/ml')

# Error handler for the ML API routes
@ml_api.errorhandler(Exception)
def handle_error(error):
    logger.exception("ML API error: %s", str(error))
    return jsonify({
        "error": str(error),
        "status": "error"
    }), 500

# Model Configuration Routes
@ml_api.route('/model-configs', methods=['GET'])
@login_required
def get_model_configs():
    """Get all model configurations for the current user."""
    user_id = current_user.id if current_user.is_authenticated else None
    
    service = MLSettingsService()
    configs = service.get_model_configurations(user_id)
    
    return jsonify({
        "status": "success",
        "configurations": configs
    })

@ml_api.route('/model-configs/<config_id>', methods=['GET'])
@login_required
def get_model_config(config_id):
    """Get a specific model configuration."""
    service = MLSettingsService()
    config = service.get_model_configuration(config_id)
    
    if not config:
        return jsonify({
            "status": "error",
            "error": f"Model configuration with ID {config_id} not found"
        }), 404
    
    return jsonify({
        "status": "success",
        "configuration": config
    })

@ml_api.route('/model-configs', methods=['POST'])
@login_required
def create_model_config():
    """Create a new model configuration."""
    data = request.json
    user_id = current_user.id if current_user.is_authenticated else None
    
    service = MLSettingsService()
    config = service.create_model_configuration(data, user_id)
    
    return jsonify({
        "status": "success",
        "configuration": config
    }), 201

@ml_api.route('/model-configs/<config_id>', methods=['PUT'])
@login_required
def update_model_config(config_id):
    """Update an existing model configuration."""
    data = request.json
    
    service = MLSettingsService()
    config = service.update_model_configuration(config_id, data)
    
    if not config:
        return jsonify({
            "status": "error",
            "error": f"Model configuration with ID {config_id} not found"
        }), 404
    
    return jsonify({
        "status": "success",
        "configuration": config
    })

@ml_api.route('/model-configs/<config_id>', methods=['DELETE'])
@login_required
def delete_model_config(config_id):
    """Delete a model configuration."""
    service = MLSettingsService()
    result = service.delete_model_configuration(config_id)
    
    if not result:
        return jsonify({
            "status": "error",
            "error": f"Model configuration with ID {config_id} not found"
        }), 404
    
    return jsonify({
        "status": "success",
        "message": f"Model configuration with ID {config_id} deleted"
    })

# Metric Configuration Routes
@ml_api.route('/metric-configs', methods=['GET'])
@login_required
def get_metric_configs():
    """Get all metric configurations for the current user."""
    user_id = current_user.id if current_user.is_authenticated else None
    
    service = MLSettingsService()
    configs = service.get_metric_configurations(user_id)
    
    return jsonify({
        "status": "success",
        "configurations": configs
    })

@ml_api.route('/metric-configs', methods=['POST'])
@login_required
def create_metric_config():
    """Create a new metric configuration."""
    data = request.json
    user_id = current_user.id if current_user.is_authenticated else None
    
    service = MLSettingsService()
    config = service.create_metric_configuration(data, user_id)
    
    return jsonify({
        "status": "success",
        "configuration": config
    }), 201

# Experiment Routes
@ml_api.route('/experiments', methods=['GET'])
@login_required
def get_experiments():
    """Get all experiments for the current user."""
    user_id = current_user.id if current_user.is_authenticated else None
    
    service = MLExperimentService()
    experiments = service.get_experiments(user_id)
    
    return jsonify({
        "status": "success",
        "experiments": experiments
    })

@ml_api.route('/experiments/<experiment_id>', methods=['GET'])
@login_required
def get_experiment(experiment_id):
    """Get a specific experiment."""
    service = MLExperimentService()
    experiment = service.get_experiment(experiment_id)
    
    if not experiment:
        return jsonify({
            "status": "error",
            "error": f"Experiment with ID {experiment_id} not found"
        }), 404
    
    return jsonify({
        "status": "success",
        "experiment": experiment
    })

@ml_api.route('/experiments', methods=['POST'])
@login_required
def create_experiment():
    """Create a new experiment."""
    data = request.json
    user_id = current_user.id if current_user.is_authenticated else None
    
    service = MLExperimentService()
    experiment = service.create_experiment(
        name=data.get('name', 'New Experiment'),
        description=data.get('description'),
        model_config_id=data.get('model_config_id'),
        metric_config_id=data.get('metric_config_id'),
        user_id=user_id
    )
    
    return jsonify({
        "status": "success",
        "experiment": experiment
    }), 201

@ml_api.route('/experiments/<experiment_id>/iterations', methods=['GET'])
@login_required
def get_experiment_iterations(experiment_id):
    """Get all iterations for an experiment."""
    service = MLExperimentService()
    iterations = service.get_experiment_iterations(experiment_id)
    
    return jsonify({
        "status": "success",
        "iterations": iterations
    })

# 5-API Workflow Routes
@ml_api.route('/workflow/run', methods=['POST'])
@login_required
def run_workflow():
    """Run the 5-API workflow."""
    data = request.json
    experiment_id = data.get('experiment_id')
    
    if not experiment_id:
        return jsonify({
            "status": "error",
            "error": "Experiment ID is required"
        }), 400
    
    # Get system and output prompts
    system_prompt = data.get('system_prompt')
    output_prompt = data.get('output_prompt')
    
    if not system_prompt or not output_prompt:
        return jsonify({
            "status": "error",
            "error": "System prompt and output prompt are required"
        }), 400
    
    # Get examples
    examples = data.get('examples', [])
    if not examples:
        return jsonify({
            "status": "error",
            "error": "Examples are required"
        }), 400
    
    # Initialize workflow
    try:
        # This would normally use actual clients from the app context
        from app.llm_client import LLMClient
        from app.evaluator import Evaluator
        from app.data_module import DataModule
        
        workflow = FiveAPIWorkflow(
            experiment_id=experiment_id,
            llm_client=LLMClient(),
            evaluator_client=Evaluator(),
            data_module=DataModule(),
            max_iterations=data.get('max_iterations', 5),
            target_metric=data.get('target_metric', 'exact_match'),
            target_threshold=data.get('target_threshold', 0.9),
            early_stopping_patience=data.get('early_stopping_patience', 2)
        )
        
        # Run the workflow
        results = workflow.run(system_prompt, output_prompt, examples)
        
        return jsonify({
            "status": "success",
            "results": results
        })
    
    except Exception as e:
        logger.exception("Error running 5-API workflow")
        return jsonify({
            "status": "error",
            "error": f"Error running workflow: {str(e)}"
        }), 500

# Meta-Learning Model Routes
@ml_api.route('/meta-learning-models', methods=['GET'])
@login_required
def get_meta_learning_models():
    """Get all meta-learning models for the current user."""
    user_id = current_user.id if current_user.is_authenticated else None
    
    service = MetaLearningService()
    models = service.get_meta_learning_models(user_id)
    
    return jsonify({
        "status": "success",
        "models": models
    })

@ml_api.route('/meta-learning-models', methods=['POST'])
@login_required
def create_meta_learning_model():
    """Create a new meta-learning model."""
    data = request.json
    user_id = current_user.id if current_user.is_authenticated else None
    
    service = MetaLearningService()
    model = service.create_meta_learning_model(data, user_id)
    
    return jsonify({
        "status": "success",
        "model": model
    }), 201

@ml_api.route('/meta-learning-models/<model_id>', methods=['PUT'])
@login_required
def update_meta_learning_model(model_id):
    """Update an existing meta-learning model."""
    data = request.json
    
    service = MetaLearningService()
    model = service.update_meta_learning_model(model_id, data)
    
    if not model:
        return jsonify({
            "status": "error",
            "error": f"Meta-learning model with ID {model_id} not found"
        }), 404
    
    return jsonify({
        "status": "success",
        "model": model
    })

# RL Model Routes
@ml_api.route('/rl-models', methods=['GET'])
@login_required
def get_rl_models():
    """Get all RL models for the current user."""
    user_id = current_user.id if current_user.is_authenticated else None
    
    service = RLModelService()
    models = service.get_rl_models(user_id)
    
    return jsonify({
        "status": "success",
        "models": models
    })

@ml_api.route('/rl-models', methods=['POST'])
@login_required
def create_rl_model():
    """Create a new RL model."""
    data = request.json
    user_id = current_user.id if current_user.is_authenticated else None
    
    service = RLModelService()
    model = service.create_rl_model(data, user_id)
    
    return jsonify({
        "status": "success",
        "model": model
    }), 201

# Metrics Summary API
@ml_api.route('/metrics-summary', methods=['GET'])
@login_required
def get_metrics_summary():
    """Get a summary of metrics across experiments."""
    user_id = current_user.id if current_user.is_authenticated else None
    
    service = MLExperimentService()
    experiments = service.get_experiments(user_id)
    
    # Calculate summary metrics
    completed_experiments = [exp for exp in experiments if exp.get('status') == 'completed']
    
    if not completed_experiments:
        return jsonify({
            "status": "success",
            "training_accuracy": 0.0,
            "validation_accuracy": 0.0,
            "improvement_percentage": 0
        })
    
    # Extract metrics from the most recent completed experiment
    most_recent = max(completed_experiments, key=lambda x: x.get('updated_at', ''))
    result_data = most_recent.get('result_data', {})
    final_metrics = result_data.get('final_metrics', {})
    
    training_accuracy = final_metrics.get('training_score', 0.0)
    validation_accuracy = final_metrics.get('validation_score', 0.0)
    
    # Calculate improvement percentage
    iterations = result_data.get('iterations', [])
    if iterations:
        first_iteration = iterations[0]
        primary_score = first_iteration.get('primary_metrics', {}).get(
            'exact_match', 0.0
        )
        
        if primary_score > 0:
            improvement_percentage = int(100 * (training_accuracy - primary_score) / primary_score)
        else:
            improvement_percentage = 0
    else:
        improvement_percentage = 0
    
    return jsonify({
        "status": "success",
        "training_accuracy": training_accuracy,
        "validation_accuracy": validation_accuracy,
        "improvement_percentage": improvement_percentage
    })