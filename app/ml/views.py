"""
Flask views for ML settings and dashboards in the ATLAS platform.

This module defines routes for:
- ML settings dashboard
- Model configuration management
- Metric configuration management
- Experiment visualization
- Meta-learning management
"""

from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
from flask_login import login_required, current_user
import logging
from typing import Dict, Any, List, Optional

from app.ml.services import (
    MLSettingsService, 
    MLExperimentService, 
    MetaLearningService,
    RLModelService
)

logger = logging.getLogger(__name__)

ml_dashboard = Blueprint('ml_dashboard', __name__, url_prefix='/ml-dashboard')

# ML Settings Dashboard
@ml_dashboard.route('/')
@login_required
def index():
    """Render the ML settings dashboard."""
    return render_template('ml/index.html', title="ML Dashboard")

# Model Configuration Views
@ml_dashboard.route('/model-configurations')
@login_required
def model_configurations():
    """Render the model configurations page."""
    user_id = current_user.id if current_user.is_authenticated else None
    
    service = MLSettingsService()
    configurations = service.get_model_configurations(user_id)
    
    return render_template(
        'ml/model_configurations.html',
        configurations=configurations,
        title="ML Model Configurations"
    )

@ml_dashboard.route('/model-configurations/create', methods=['GET', 'POST'])
@login_required
def create_model_configuration():
    """Create a new model configuration."""
    if request.method == 'POST':
        user_id = current_user.id if current_user.is_authenticated else None
        
        config_data = {
            "name": request.form.get('name'),
            "primary_model": request.form.get('primary_model'),
            "optimizer_model": request.form.get('optimizer_model'),
            "temperature": float(request.form.get('temperature', 0.0)),
            "max_tokens": int(request.form.get('max_tokens', 1024)),
            "top_p": float(request.form.get('top_p', 1.0)),
            "top_k": int(request.form.get('top_k', 40)),
            "is_default": 'is_default' in request.form
        }
        
        service = MLSettingsService()
        service.create_model_configuration(config_data, user_id)
        
        flash("Model configuration created successfully.", "success")
        return redirect(url_for('ml_dashboard.model_configurations'))
    
    return render_template(
        'ml/create_model_configuration.html',
        title="Create Model Configuration"
    )

@ml_dashboard.route('/model-configurations/<config_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_model_configuration(config_id):
    """Edit an existing model configuration."""
    service = MLSettingsService()
    config = service.get_model_configuration(config_id)
    
    if not config:
        flash("Model configuration not found.", "danger")
        return redirect(url_for('ml_dashboard.model_configurations'))
    
    if request.method == 'POST':
        config_data = {
            "name": request.form.get('name'),
            "primary_model": request.form.get('primary_model'),
            "optimizer_model": request.form.get('optimizer_model'),
            "temperature": float(request.form.get('temperature', 0.0)),
            "max_tokens": int(request.form.get('max_tokens', 1024)),
            "top_p": float(request.form.get('top_p', 1.0)),
            "top_k": int(request.form.get('top_k', 40)),
            "is_default": 'is_default' in request.form
        }
        
        service.update_model_configuration(config_id, config_data)
        
        flash("Model configuration updated successfully.", "success")
        return redirect(url_for('ml_dashboard.model_configurations'))
    
    return render_template(
        'ml/edit_model_configuration.html',
        config=config,
        title="Edit Model Configuration"
    )

@ml_dashboard.route('/model-configurations/<config_id>/delete', methods=['POST'])
@login_required
def delete_model_configuration(config_id):
    """Delete a model configuration."""
    service = MLSettingsService()
    result = service.delete_model_configuration(config_id)
    
    if result:
        flash("Model configuration deleted successfully.", "success")
    else:
        flash("Failed to delete model configuration.", "danger")
    
    return redirect(url_for('ml_dashboard.model_configurations'))

# Metric Configuration Views
@ml_dashboard.route('/metric-configurations')
@login_required
def metric_configurations():
    """Render the metric configurations page."""
    user_id = current_user.id if current_user.is_authenticated else None
    
    service = MLSettingsService()
    configurations = service.get_metric_configurations(user_id)
    
    return render_template(
        'ml/metric_configurations.html',
        configurations=configurations,
        title="ML Metric Configurations"
    )

@ml_dashboard.route('/metric-configurations/create', methods=['GET', 'POST'])
@login_required
def create_metric_configuration():
    """Create a new metric configuration."""
    if request.method == 'POST':
        user_id = current_user.id if current_user.is_authenticated else None
        
        # Process the form data
        metrics = request.form.getlist('metrics')
        weights = {}
        for metric in metrics:
            weight_key = f'weight_{metric}'
            if weight_key in request.form:
                try:
                    weights[metric] = float(request.form.get(weight_key, 1.0))
                except ValueError:
                    weights[metric] = 1.0
        
        config_data = {
            "name": request.form.get('name'),
            "metrics": metrics,
            "metric_weights": weights,
            "target_threshold": float(request.form.get('target_threshold', 0.8))
        }
        
        service = MLSettingsService()
        service.create_metric_configuration(config_data, user_id)
        
        flash("Metric configuration created successfully.", "success")
        return redirect(url_for('ml_dashboard.metric_configurations'))
    
    # Available metrics for the form
    available_metrics = [
        {"id": "exact_match", "name": "Exact Match"},
        {"id": "semantic_similarity", "name": "Semantic Similarity"},
        {"id": "keyword_match", "name": "Keyword Match"},
        {"id": "bleu", "name": "BLEU Score"},
        {"id": "rouge", "name": "ROUGE Score"},
        {"id": "f1_score", "name": "F1 Score"}
    ]
    
    return render_template(
        'ml/create_metric_configuration.html',
        available_metrics=available_metrics,
        title="Create Metric Configuration"
    )

# Experiment Views
@ml_dashboard.route('/experiments')
@login_required
def experiments():
    """Render the experiments page."""
    user_id = current_user.id if current_user.is_authenticated else None
    
    service = MLExperimentService()
    experiments = service.get_experiments(user_id)
    
    return render_template(
        'ml/experiments.html',
        experiments=experiments,
        title="ML Experiments"
    )

@ml_dashboard.route('/experiments/create', methods=['GET', 'POST'])
@login_required
def create_experiment():
    """Create a new experiment."""
    if request.method == 'POST':
        user_id = current_user.id if current_user.is_authenticated else None
        
        service = MLExperimentService()
        experiment = service.create_experiment(
            name=request.form.get('name'),
            description=request.form.get('description'),
            model_config_id=request.form.get('model_config_id'),
            metric_config_id=request.form.get('metric_config_id'),
            user_id=user_id
        )
        
        flash("Experiment created successfully.", "success")
        return redirect(url_for('ml_dashboard.experiments'))
    
    # Get available model and metric configurations
    ml_settings_service = MLSettingsService()
    model_configs = ml_settings_service.get_model_configurations()
    metric_configs = ml_settings_service.get_metric_configurations()
    
    return render_template(
        'ml/create_experiment.html',
        model_configs=model_configs,
        metric_configs=metric_configs,
        title="Create Experiment"
    )

@ml_dashboard.route('/experiments/<experiment_id>')
@login_required
def view_experiment(experiment_id):
    """View a specific experiment."""
    service = MLExperimentService()
    experiment = service.get_experiment(experiment_id)
    
    if not experiment:
        flash("Experiment not found.", "danger")
        return redirect(url_for('ml_dashboard.experiments'))
    
    iterations = service.get_experiment_iterations(experiment_id)
    
    return render_template(
        'ml/view_experiment.html',
        experiment=experiment,
        iterations=iterations,
        title=f"Experiment: {experiment.get('name', 'Unknown')}"
    )

@ml_dashboard.route('/experiments/<experiment_id>/run', methods=['GET', 'POST'])
@login_required
def run_experiment(experiment_id):
    """Run an experiment with the 5-API workflow."""
    service = MLExperimentService()
    experiment = service.get_experiment(experiment_id)
    
    if not experiment:
        flash("Experiment not found.", "danger")
        return redirect(url_for('ml_dashboard.experiments'))
    
    if request.method == 'POST':
        # In a real implementation, this would start a background task
        # For now, we'll just update the status
        service.update_experiment_status(experiment_id, "running")
        
        flash("Experiment started successfully. Check back later for results.", "success")
        return redirect(url_for('ml_dashboard.view_experiment', experiment_id=experiment_id))
    
    return render_template(
        'ml/run_experiment.html',
        experiment=experiment,
        title=f"Run Experiment: {experiment.get('name', 'Unknown')}"
    )

# Meta-Learning Views
@ml_dashboard.route('/meta-learning')
@login_required
def meta_learning():
    """Render the meta-learning page."""
    user_id = current_user.id if current_user.is_authenticated else None
    
    service = MetaLearningService()
    models = service.get_meta_learning_models(user_id)
    
    return render_template(
        'ml/meta_learning.html',
        models=models,
        title="Meta-Learning Models"
    )

@ml_dashboard.route('/meta-learning/create', methods=['GET', 'POST'])
@login_required
def create_meta_learning_model():
    """Create a new meta-learning model."""
    if request.method == 'POST':
        user_id = current_user.id if current_user.is_authenticated else None
        
        # Process the form data
        hyperparameters = {}
        for key in request.form:
            if key.startswith('hyper_'):
                param_name = key[6:]  # Remove 'hyper_' prefix
                try:
                    value = float(request.form.get(key))
                except ValueError:
                    value = request.form.get(key)
                hyperparameters[param_name] = value
        
        feature_config = {
            "selected_features": request.form.getlist('features'),
            "feature_extraction": request.form.get('feature_extraction', 'manual'),
            "normalization": request.form.get('normalization', 'standard')
        }
        
        model_data = {
            "name": request.form.get('name'),
            "model_type": request.form.get('model_type'),
            "hyperparameters": hyperparameters,
            "feature_config": feature_config,
            "is_active": 'is_active' in request.form
        }
        
        service = MetaLearningService()
        service.create_meta_learning_model(model_data, user_id)
        
        flash("Meta-learning model created successfully.", "success")
        return redirect(url_for('ml_dashboard.meta_learning'))
    
    # Available model types and features for the form
    model_types = [
        {"id": "lightgbm", "name": "LightGBM"},
        {"id": "xgboost", "name": "XGBoost"},
        {"id": "random_forest", "name": "Random Forest"},
        {"id": "neural_network", "name": "Neural Network"}
    ]
    
    available_features = [
        {"id": "prompt_length", "name": "Prompt Length"},
        {"id": "output_length", "name": "Output Length"},
        {"id": "keyword_density", "name": "Keyword Density"},
        {"id": "semantic_complexity", "name": "Semantic Complexity"},
        {"id": "instruction_clarity", "name": "Instruction Clarity"},
        {"id": "example_count", "name": "Example Count"},
        {"id": "token_count", "name": "Token Count"}
    ]
    
    return render_template(
        'ml/create_meta_learning_model.html',
        model_types=model_types,
        available_features=available_features,
        title="Create Meta-Learning Model"
    )

# RL Model Views
@ml_dashboard.route('/rl-models')
@login_required
def rl_models():
    """Render the RL models page."""
    user_id = current_user.id if current_user.is_authenticated else None
    
    service = RLModelService()
    models = service.get_rl_models(user_id)
    
    return render_template(
        'ml/rl_models.html',
        models=models,
        title="Reinforcement Learning Models"
    )