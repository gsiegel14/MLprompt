
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from sqlalchemy.orm import Session
from src.app.utils.ml_settings_service import MLSettingsService
from src.app.models.ml_models import ModelConfiguration, MetricConfiguration, MetaLearningConfiguration
import logging
from src.app.db.database import get_db
import json

logger = logging.getLogger(__name__)

ml_settings_bp = Blueprint('ml_settings', __name__, url_prefix='/ml-settings')

@ml_settings_bp.route('/')
def index():
    """Render the ML settings dashboard"""
    return render_template('dashboard/ml_settings/index.html')

# Model Configuration Routes
@ml_settings_bp.route('/models')
def model_configurations():
    """Render the model configurations page"""
    try:
        db = next(get_db())
        service = MLSettingsService(db)
        configurations = service.get_model_configurations()
        return render_template('dashboard/ml_settings/models.html', configurations=configurations)
    except Exception as e:
        logger.error(f"Error loading model configurations: {str(e)}")
        flash(f"Error loading model configurations: {str(e)}", "danger")
        return render_template('dashboard/ml_settings/models.html', configurations=[])

@ml_settings_bp.route('/models/create', methods=['GET', 'POST'])
def create_model_configuration():
    """Create a new model configuration"""
    if request.method == 'POST':
        try:
            db = next(get_db())
            service = MLSettingsService(db)
            
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
            
            service.create_model_configuration(config_data)
            flash("Model configuration created successfully!", "success")
            return redirect(url_for('ml_settings.model_configurations'))
        except Exception as e:
            logger.error(f"Error creating model configuration: {str(e)}")
            flash(f"Error creating model configuration: {str(e)}", "danger")
    
    return render_template('dashboard/ml_settings/create_model.html')

@ml_settings_bp.route('/models/edit/<config_id>', methods=['GET', 'POST'])
def edit_model_configuration(config_id):
    """Edit a model configuration"""
    db = next(get_db())
    service = MLSettingsService(db)
    
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
                "is_default": 'is_default' in request.form
            }
            
            service.update_model_configuration(config_id, config_data)
            flash("Model configuration updated successfully!", "success")
            return redirect(url_for('ml_settings.model_configurations'))
        except Exception as e:
            logger.error(f"Error updating model configuration: {str(e)}")
            flash(f"Error updating model configuration: {str(e)}", "danger")
    
    config = service.get_model_configuration(config_id)
    if not config:
        flash("Configuration not found", "danger")
        return redirect(url_for('ml_settings.model_configurations'))
    
    return render_template('dashboard/ml_settings/create_model.html', config=config, edit_mode=True)

@ml_settings_bp.route('/models/delete/<config_id>')
def delete_model_configuration(config_id):
    """Delete a model configuration"""
    try:
        db = next(get_db())
        service = MLSettingsService(db)
        
        if service.delete_model_configuration(config_id):
            flash("Model configuration deleted successfully!", "success")
        else:
            flash("Configuration not found", "danger")
    except Exception as e:
        logger.error(f"Error deleting model configuration: {str(e)}")
        flash(f"Error deleting model configuration: {str(e)}", "danger")
    
    return redirect(url_for('ml_settings.model_configurations'))

# Metric Configuration Routes
@ml_settings_bp.route('/metrics')
def metric_configurations():
    """Render the metric configurations page"""
    try:
        db = next(get_db())
        service = MLSettingsService(db)
        configurations = service.get_metric_configurations()
        return render_template('dashboard/ml_settings/metrics.html', configurations=configurations)
    except Exception as e:
        logger.error(f"Error loading metric configurations: {str(e)}")
        flash(f"Error loading metric configurations: {str(e)}", "danger")
        return render_template('dashboard/ml_settings/metrics.html', configurations=[])

# Meta-Learning Configuration Routes
@ml_settings_bp.route('/meta-learning')
def meta_learning_configurations():
    """Render the meta-learning configurations page"""
    try:
        db = next(get_db())
        service = MLSettingsService(db)
        configurations = service.get_meta_learning_configurations()
        return render_template('dashboard/ml_settings/meta_learning.html', configurations=configurations)
    except Exception as e:
        logger.error(f"Error loading meta-learning configurations: {str(e)}")
        flash(f"Error loading meta-learning configurations: {str(e)}", "danger")
        return render_template('dashboard/ml_settings/meta_learning.html', configurations=[])

@ml_settings_bp.route('/meta-learning/train', methods=['POST'])
def train_meta_model():
    """Train a meta-learning model"""
    try:
        config_id = request.form.get('config_id')
        
        if not config_id:
            return jsonify({"status": "error", "message": "No configuration ID provided"})
        
        # Here we would start background processing with a task queue like Celery
        # For now, we'll simulate a task ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # In a real implementation, you would use:
        # from src.app.tasks.ml_tasks import train_meta_model_task
        # task = train_meta_model_task.delay(config_id)
        # task_id = task.id
        
        return jsonify({"status": "training_started", "task_id": task_id})
    except Exception as e:
        logger.error(f"Error starting meta-model training: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

# Experiment Visualization Routes
@ml_settings_bp.route('/visualization')
def experiment_visualization():
    """Render the experiment visualization page"""
    try:
        # In a real implementation, you would fetch experiments from your database
        # For now, we'll create some dummy data
        experiments = [
            {"id": "exp1", "name": "Experiment A"},
            {"id": "exp2", "name": "Experiment B"},
            {"id": "exp3", "name": "Experiment C"}
        ]
        
        return render_template('dashboard/ml_settings/visualization.html', experiments=experiments)
    except Exception as e:
        logger.error(f"Error loading experiment visualization: {str(e)}")
        flash(f"Error loading experiment visualization: {str(e)}", "danger")
        return render_template('dashboard/ml_settings/visualization.html', experiments=[])
