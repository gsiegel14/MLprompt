"""
Frontend views for the ML functionality in the ATLAS platform.

This module provides routes for:
1. ML Dashboard
2. Prompt Management
3. Experiment Monitoring
4. Model Configuration 
"""

import os
import logging
import json
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_required, current_user

from app.ml.services import (
    MLSettingsService as MLConfigService,
    MLExperimentService,
    MetaLearningService,
    RLModelService
)

logger = logging.getLogger(__name__)

# Create the blueprint
ml_views = Blueprint('ml_views', __name__, url_prefix='/ml')

# ML Dashboard
@ml_views.route('/', methods=['GET'])
@login_required
def index():
    """Render the ML Dashboard."""
    # Get metrics summary
    service = MLExperimentService()
    user_id = current_user.id if current_user.is_authenticated else None
    
    # Get recent experiments
    recent_experiments = service.get_experiments(user_id, limit=5)
    
    # Get metrics
    metrics = {
        'training_accuracy': 0.0,
        'validation_accuracy': 0.0,
        'improvement_percentage': 0
    }
    
    # Calculate metrics from completed experiments
    completed_experiments = [exp for exp in recent_experiments if exp.get('status') == 'completed']
    if completed_experiments:
        # Get the most recent completed experiment
        most_recent = max(completed_experiments, key=lambda x: x.get('updated_at', ''))
        result_data = most_recent.get('result_data', {})
        final_metrics = result_data.get('final_metrics', {})
        
        metrics['training_accuracy'] = final_metrics.get('training_score', 0.0)
        metrics['validation_accuracy'] = final_metrics.get('validation_score', 0.0)
        
        # Calculate improvement percentage
        iterations = result_data.get('iterations', [])
        if iterations:
            first_iteration = iterations[0]
            primary_score = first_iteration.get('primary_metrics', {}).get(
                'exact_match', 0.0
            )
            
            if primary_score > 0:
                metrics['improvement_percentage'] = int(100 * (metrics['training_accuracy'] - primary_score) / primary_score)
    
    # Get models
    ml_service = MetaLearningService()
    rl_service = RLModelService()
    
    ml_models = ml_service.get_meta_learning_models(user_id)
    rl_models = rl_service.get_rl_models(user_id)
    
    return render_template(
        'ml/index.html',
        experiments=recent_experiments,
        metrics=metrics,
        ml_models=ml_models,
        rl_models=rl_models,
        active_page='ml_dashboard'
    )

# Experiment Management
@ml_views.route('/experiments', methods=['GET'])
@login_required
def experiments():
    """Render the experiments page."""
    service = MLExperimentService()
    user_id = current_user.id if current_user.is_authenticated else None
    
    experiments = service.get_experiments(user_id)
    
    return render_template(
        'ml/experiments.html',
        experiments=experiments,
        active_page='ml_experiments'
    )

@ml_views.route('/experiments/<experiment_id>', methods=['GET'])
@login_required
def experiment_detail(experiment_id):
    """Render the experiment detail page."""
    service = MLExperimentService()
    
    experiment = service.get_experiment(experiment_id)
    if not experiment:
        flash('Experiment not found', 'error')
        return redirect(url_for('ml_views.experiments'))
    
    iterations = service.get_experiment_iterations(experiment_id)
    
    return render_template(
        'ml/experiment_detail.html',
        experiment=experiment,
        iterations=iterations,
        active_page='ml_experiments'
    )

@ml_views.route('/experiments/new', methods=['GET'])
@login_required
def new_experiment():
    """Render the new experiment page."""
    return render_template(
        'ml/new_experiment.html',
        active_page='ml_experiments'
    )

# Model Configurations
@ml_views.route('/models/configurations', methods=['GET'])
@login_required
def model_configurations():
    """Render the model configurations page."""
    service = MLConfigService()
    user_id = current_user.id if current_user.is_authenticated else None
    
    model_configs = service.get_model_configurations(user_id)
    metric_configs = service.get_metric_configurations(user_id)
    
    return render_template(
        'ml/model_configurations.html',
        model_configs=model_configs,
        metric_configs=metric_configs,
        active_page='ml_configurations'
    )

# Metric Configurations
@ml_views.route('/metrics/configurations', methods=['GET'])
@login_required
def metric_configurations():
    """Render the metric configurations page."""
    service = MLConfigService()
    user_id = current_user.id if current_user.is_authenticated else None
    
    metric_configs = service.get_metric_configurations(user_id)
    
    return render_template(
        'ml/metric_configurations.html',
        metric_configs=metric_configs,
        active_page='ml_metric_configurations'
    )

# Meta-Learning Models
@ml_views.route('/models/meta-learning', methods=['GET'])
@login_required
def meta_learning_models():
    """Render the meta-learning models page."""
    service = MetaLearningService()
    user_id = current_user.id if current_user.is_authenticated else None
    
    models = service.get_meta_learning_models(user_id)
    
    return render_template(
        'ml/meta_learning_models.html',
        models=models,
        active_page='ml_meta_learning'
    )

@ml_views.route('/models/meta-learning/<model_id>', methods=['GET'])
@login_required
def meta_learning_model_detail(model_id):
    """Render the meta-learning model detail page."""
    service = MetaLearningService()
    
    model = service.get_meta_learning_model(model_id)
    if not model:
        flash('Model not found', 'error')
        return redirect(url_for('ml_views.meta_learning_models'))
    
    return render_template(
        'ml/meta_learning_model_detail.html',
        model=model,
        active_page='ml_meta_learning'
    )

# RL Models
@ml_views.route('/models/rl', methods=['GET'])
@login_required
def rl_models():
    """Render the RL models page."""
    service = RLModelService()
    user_id = current_user.id if current_user.is_authenticated else None
    
    models = service.get_rl_models(user_id)
    
    return render_template(
        'ml/rl_models.html',
        models=models,
        active_page='ml_rl_models'
    )

@ml_views.route('/models/rl/<model_id>', methods=['GET'])
@login_required
def rl_model_detail(model_id):
    """Render the RL model detail page."""
    service = RLModelService()
    
    model = service.get_rl_model(model_id)
    if not model:
        flash('Model not found', 'error')
        return redirect(url_for('ml_views.rl_models'))
    
    return render_template(
        'ml/rl_model_detail.html',
        model=model,
        active_page='ml_rl_models'
    )

# Prompt Management
@ml_views.route('/prompts', methods=['GET'])
@login_required
def prompts():
    """Render the prompts page."""
    service = MLExperimentService()
    user_id = current_user.id if current_user.is_authenticated else None
    
    # Get recent experiments for their prompts
    experiments = service.get_experiments(user_id)
    
    # Extract prompts from experiments
    original_prompts = []
    evaluator_prompts = []
    optimizer_prompts = []
    final_prompts = []
    
    for exp in experiments:
        if exp.get('status') == 'completed':
            # Original prompt is stored in the experiment
            original_prompts.append({
                'experiment_id': exp.get('id'),
                'name': exp.get('name', 'Unnamed'),
                'system_prompt': exp.get('system_prompt', ''),
                'output_prompt': exp.get('output_prompt', ''),
                'created_at': exp.get('created_at')
            })
            
            # Extract other prompts from iterations
            iterations = service.get_experiment_iterations(exp.get('id'))
            if iterations:
                # Evaluator and Optimizer prompts are in metadata
                metadata = exp.get('metadata', {})
                if 'evaluator_prompt' in metadata:
                    evaluator_prompts.append({
                        'experiment_id': exp.get('id'),
                        'name': f"Evaluator for {exp.get('name', 'Unnamed')}",
                        'system_prompt': metadata.get('evaluator_prompt', {}).get('system_prompt', ''),
                        'output_prompt': metadata.get('evaluator_prompt', {}).get('output_prompt', ''),
                        'created_at': exp.get('created_at')
                    })
                
                if 'optimizer_prompt' in metadata:
                    optimizer_prompts.append({
                        'experiment_id': exp.get('id'),
                        'name': f"Optimizer for {exp.get('name', 'Unnamed')}",
                        'system_prompt': metadata.get('optimizer_prompt', {}).get('system_prompt', ''),
                        'output_prompt': metadata.get('optimizer_prompt', {}).get('output_prompt', ''),
                        'created_at': exp.get('created_at')
                    })
                
                # Final prompt is in the last iteration
                if iterations:
                    last_iteration = iterations[-1]
                    final_prompts.append({
                        'experiment_id': exp.get('id'),
                        'name': f"Final for {exp.get('name', 'Unnamed')}",
                        'system_prompt': last_iteration.get('system_prompt', ''),
                        'output_prompt': last_iteration.get('output_prompt', ''),
                        'created_at': last_iteration.get('created_at', exp.get('created_at'))
                    })
    
    return render_template(
        'ml/prompts.html',
        original_prompts=original_prompts,
        evaluator_prompts=evaluator_prompts,
        optimizer_prompts=optimizer_prompts,
        final_prompts=final_prompts,
        active_page='ml_prompts'
    )

# Workflow Visualization
@ml_views.route('/workflow', methods=['GET'])
@login_required
def workflow():
    """Render the workflow visualization page."""
    return render_template(
        'ml/workflow.html',
        active_page='ml_workflow'
    )

# Settings
@ml_views.route('/settings', methods=['GET'])
@login_required
def settings():
    """Render the ML settings page."""
    return render_template(
        'ml/settings.html',
        active_page='ml_settings'
    )