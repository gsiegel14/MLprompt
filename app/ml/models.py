"""
ML-related database models for the ATLAS platform.
These models store ML configurations, experiment data, and meta-learning information.
"""

from flask_sqlalchemy import SQLAlchemy
from app import db
from datetime import datetime
import uuid

class ModelConfiguration(db.Model):
    """Model to store LLM configuration parameters."""
    __tablename__ = "model_configurations"
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(255), nullable=False)
    primary_model = db.Column(db.String(255), nullable=False)  # e.g., "gemini-1.5-flash"
    optimizer_model = db.Column(db.String(255), nullable=False)  # e.g., "gemini-1.5-pro"
    temperature = db.Column(db.Float, default=0.0)
    max_tokens = db.Column(db.Integer, default=1024)
    top_p = db.Column(db.Float, default=1.0)
    top_k = db.Column(db.Integer, default=40)
    is_default = db.Column(db.Boolean, default=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = db.relationship("User", backref=db.backref("model_configurations", lazy=True))

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'primary_model': self.primary_model,
            'optimizer_model': self.optimizer_model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'is_default': self.is_default,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class MetricConfiguration(db.Model):
    """Model to store evaluation metric configurations."""
    __tablename__ = "metric_configurations"
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(255), nullable=False)
    metrics = db.Column(db.JSON, nullable=False)  # e.g., ["exact_match", "bleu"]
    metric_weights = db.Column(db.JSON, default=lambda: {})  # e.g., {"exact_match": 0.7, "bleu": 0.3}
    target_threshold = db.Column(db.Float, default=0.8)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = db.relationship("User", backref=db.backref("metric_configurations", lazy=True))

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'metrics': self.metrics,
            'metric_weights': self.metric_weights,
            'target_threshold': self.target_threshold,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class MLExperiment(db.Model):
    """Model to store ML experiment results."""
    __tablename__ = "ml_experiments"
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    model_config_id = db.Column(db.String(36), db.ForeignKey("model_configurations.id"))
    metric_config_id = db.Column(db.String(36), db.ForeignKey("metric_configurations.id"))
    status = db.Column(db.String(50), default="created")  # created, running, completed, failed
    result_data = db.Column(db.JSON, default=lambda: {})
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = db.relationship("User", backref=db.backref("ml_experiments", lazy=True))
    model_config = db.relationship("ModelConfiguration")
    metric_config = db.relationship("MetricConfiguration")
    iterations = db.relationship("MLExperimentIteration", back_populates="experiment", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'status': self.status,
            'result_data': self.result_data,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'model_config': self.model_config.to_dict() if self.model_config else None,
            'metric_config': self.metric_config.to_dict() if self.metric_config else None
        }

class MLExperimentIteration(db.Model):
    """Model to store individual ML experiment iterations."""
    __tablename__ = "ml_experiment_iterations"
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = db.Column(db.String(36), db.ForeignKey("ml_experiments.id"), nullable=False)
    iteration_number = db.Column(db.Integer, nullable=False)
    system_prompt = db.Column(db.Text, nullable=True)
    output_prompt = db.Column(db.Text, nullable=True)
    metrics = db.Column(db.JSON, default=lambda: {})
    training_accuracy = db.Column(db.Float, nullable=True)
    validation_accuracy = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    experiment = db.relationship("MLExperiment", back_populates="iterations")

    def to_dict(self):
        return {
            'id': self.id,
            'experiment_id': self.experiment_id,
            'iteration_number': self.iteration_number,
            'system_prompt': self.system_prompt,
            'output_prompt': self.output_prompt,
            'metrics': self.metrics,
            'training_accuracy': self.training_accuracy,
            'validation_accuracy': self.validation_accuracy,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class MetaLearningModel(db.Model):
    """Model to store meta-learning model configurations and data."""
    __tablename__ = "meta_learning_models"
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(255), nullable=False)
    model_type = db.Column(db.String(50), default="lightgbm")  # lightgbm, xgboost, etc.
    hyperparameters = db.Column(db.JSON, default=lambda: {})
    feature_config = db.Column(db.JSON, default=lambda: {})
    model_path = db.Column(db.String(255), nullable=True)
    metrics = db.Column(db.JSON, default=lambda: {})
    is_active = db.Column(db.Boolean, default=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = db.relationship("User", backref=db.backref("meta_learning_models", lazy=True))

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'model_type': self.model_type,
            'hyperparameters': self.hyperparameters,
            'feature_config': self.feature_config,
            'model_path': self.model_path,
            'metrics': self.metrics,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class RLModel(db.Model):
    """Model to store reinforcement learning model configurations and data."""
    __tablename__ = "rl_models"
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(255), nullable=False)
    model_type = db.Column(db.String(50), default="ppo")  # ppo, dqn, etc.
    hyperparameters = db.Column(db.JSON, default=lambda: {})
    action_space = db.Column(db.JSON, default=lambda: {})
    observation_space = db.Column(db.JSON, default=lambda: {})
    model_path = db.Column(db.String(255), nullable=True)
    metrics = db.Column(db.JSON, default=lambda: {})
    is_active = db.Column(db.Boolean, default=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = db.relationship("User", backref=db.backref("rl_models", lazy=True))

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'model_type': self.model_type,
            'hyperparameters': self.hyperparameters,
            'action_space': self.action_space,
            'observation_space': self.observation_space,
            'model_path': self.model_path,
            'metrics': self.metrics,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }