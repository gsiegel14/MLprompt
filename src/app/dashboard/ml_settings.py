
"""
ML Settings Dashboard Controller
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from sqlalchemy.orm import Session
import json
import os
from datetime import datetime

from src.app.utils.ml_settings_service import MLSettingsService

ml_settings_bp = Blueprint('ml_settings', __name__, url_prefix='/ml-settings')

def get_db():
    """Get database session from app context"""
    return current_app.extensions['db_session']

@ml_settings_bp.route('/')
def index():
    """ML settings dashboard overview"""
    return render_template('dashboard/ml_settings/index.html')

# Model Configuration Routes
@ml_settings_bp.route('/models')
def models():
    """Display all model configurations"""
    db = get_db()
    service = MLSettingsService(db)
    models = service.get_model_configs()
    return render_template('dashboard/ml_settings/models.html', models=models)

@ml_settings_bp.route('/models/create', methods=['GET', 'POST'])
def create_model():
    """Create a new model configuration"""
    if request.method == 'POST':
        db = get_db()
        service = MLSettingsService(db)
        
        try:
            config_data = {
                "name": request.form.get('name'),
                "primary_model": request.form.get('primary_model'),
                "optimizer_model": request.form.get('optimizer_model'),
                "temperature": float(request.form.get('temperature', 0.0)),
                "max_tokens": int(request.form.get('max_tokens', 1024)),
                "top_p": float(request.form.get('top_p', 1.0)),
                "top_k": int(request.form.get('top_k', 40)),
                "is_default": request.form.get('is_default') == 'on'
            }
            
            model = service.create_model_config(config_data)
            flash(f"Model configuration '{model.name}' created successfully!", "success")
            return redirect(url_for('ml_settings.models'))
        except Exception as e:
            flash(f"Error creating model configuration: {str(e)}", "error")
    
    # GET request
    return render_template('dashboard/ml_settings/create_model.html')

@ml_settings_bp.route('/models/<model_id>/edit', methods=['GET', 'POST'])
def edit_model(model_id):
    """Edit a model configuration"""
    db = get_db()
    service = MLSettingsService(db)
    model = service.get_model_config(model_id)
    
    if not model:
        flash("Model configuration not found", "error")
        return redirect(url_for('ml_settings.models'))
    
    if request.method == 'POST':
        try:
            config_data = {
                "name": request.form.get('name'),
                "primary_model": request.form.get('primary_model'),
                "optimizer_model": request.form.get('optimizer_model'),
                "temperature": float(request.form.get('temperature', 0.0)),
                "max_tokens": int(request.form.get('max_tokens', 1024)),
                "top_p": float(request.form.get('top_p', 1.0)),
                "top_k": int(request.form.get('top_k', 40)),
                "is_default": request.form.get('is_default') == 'on'
            }
            
            updated_model = service.update_model_config(model_id, config_data)
            flash(f"Model configuration '{updated_model.name}' updated successfully!", "success")
            return redirect(url_for('ml_settings.models'))
        except Exception as e:
            flash(f"Error updating model configuration: {str(e)}", "error")
    
    # GET request
    return render_template('dashboard/ml_settings/create_model.html', model=model)

@ml_settings_bp.route('/models/<model_id>/delete', methods=['POST'])
def delete_model(model_id):
    """Delete a model configuration"""
    db = get_db()
    service = MLSettingsService(db)
    
    if service.delete_model_config(model_id):
        flash("Model configuration deleted successfully", "success")
    else:
        flash("Error deleting model configuration", "error")
    
    return redirect(url_for('ml_settings.models'))

# Metric Configuration Routes
@ml_settings_bp.route('/metrics')
def metrics():
    """Display all metric configurations"""
    db = get_db()
    service = MLSettingsService(db)
    metrics = service.get_metric_configs()
    return render_template('dashboard/ml_settings/metrics.html', metrics=metrics)

@ml_settings_bp.route('/metrics/create', methods=['GET', 'POST'])
def create_metric():
    """Create a new metric configuration"""
    if request.method == 'POST':
        db = get_db()
        service = MLSettingsService(db)
        
        try:
            # Parse metrics list and weights
            metrics_list = request.form.get('metrics', '').split(',')
            metrics_list = [m.strip() for m in metrics_list if m.strip()]
            
            weights_str = request.form.get('metric_weights', '{}')
            try:
                metric_weights = json.loads(weights_str)
            except json.JSONDecodeError:
                metric_weights = {}
            
            config_data = {
                "name": request.form.get('name'),
                "metrics": metrics_list,
                "metric_weights": metric_weights,
                "target_threshold": float(request.form.get('target_threshold', 0.8))
            }
            
            metric = service.create_metric_config(config_data)
            flash(f"Metric configuration '{metric.name}' created successfully!", "success")
            return redirect(url_for('ml_settings.metrics'))
        except Exception as e:
            flash(f"Error creating metric configuration: {str(e)}", "error")
    
    # GET request
    return render_template('dashboard/ml_settings/metrics.html', mode="create")

# Meta-Learning Configuration Routes
@ml_settings_bp.route('/meta-learning')
def meta_learning():
    """Display all meta-learning configurations"""
    db = get_db()
    service = MLSettingsService(db)
    configs = service.get_meta_learning_configs()
    return render_template('dashboard/ml_settings/meta_learning.html', configs=configs)

# Visualization Routes
@ml_settings_bp.route('/visualization')
def visualization():
    """Visualization dashboard for experiments"""
    db = get_db()
    service = MLSettingsService(db)
    experiments = service.get_experiments(limit=10)
    return render_template('dashboard/ml_settings/visualization.html', experiments=experiments)

@ml_settings_bp.route('/api/experiments/<experiment_id>')
def get_experiment_data(experiment_id):
    """API endpoint to get experiment data for visualizations"""
    db = get_db()
    service = MLSettingsService(db)
    experiment = service.get_experiment(experiment_id)
    
    if not experiment:
        return jsonify({"error": "Experiment not found"}), 404
    
    # Load detailed metrics from experiment directory if available
    results_data = {}
    if experiment.results_path and os.path.exists(experiment.results_path):
        try:
            with open(os.path.join(experiment.results_path, 'metrics.json'), 'r') as f:
                results_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    return jsonify({
        "experiment": experiment.to_dict(),
        "detailed_metrics": results_data
    })
