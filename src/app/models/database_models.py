
"""
SQLAlchemy database models for the ML prompt optimization platform
"""
from sqlalchemy import Column, Integer, String, ForeignKey, Float, DateTime, JSON, Text, Boolean, Table
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from src.app.database.db import Base

def generate_uuid():
    """Generate a UUID string for primary keys"""
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    username = Column(String, nullable=False, unique=True)
    email = Column(String, nullable=False)
    api_key = Column(String, nullable=False, unique=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model_configs = relationship("ModelConfiguration", back_populates="user")
    metric_configs = relationship("MetricConfiguration", back_populates="user")
    meta_learning_configs = relationship("MetaLearningConfiguration", back_populates="user")
    experiments = relationship("Experiment", back_populates="user")

class Prompt(Base):
    __tablename__ = "prompts"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    system_prompt = Column(Text, nullable=False)
    output_prompt = Column(Text, nullable=False)
    version = Column(Integer, default=1)
    parent_id = Column(String, ForeignKey("prompts.id"), nullable=True)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    parent = relationship("Prompt", remote_side=[id], backref="versions")
    initial_experiments = relationship("Experiment", foreign_keys="Experiment.initial_prompt_id", back_populates="initial_prompt")
    best_experiments = relationship("Experiment", foreign_keys="Experiment.best_prompt_id", back_populates="best_prompt")
    metrics_records = relationship("MetricsRecord", back_populates="prompt")

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    file_path = Column(String, nullable=True)
    example_count = Column(Integer, default=0)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    
    # Relationships
    experiments = relationship("Experiment", back_populates="dataset")
    user = relationship("User")

class ModelConfiguration(Base):
    __tablename__ = "model_configurations"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    primary_model = Column(String, nullable=False)
    optimizer_model = Column(String, nullable=False)
    temperature = Column(Float, default=0.0)
    max_tokens = Column(Integer, default=1024)
    top_p = Column(Float, default=1.0)
    top_k = Column(Integer, default=40)
    is_default = Column(Boolean, default=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="model_configs")
    experiments = relationship("Experiment", back_populates="model_config")

class MetricConfiguration(Base):
    __tablename__ = "metric_configurations"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    metrics = Column(JSON, nullable=False)
    metric_weights = Column(JSON, default={})
    target_threshold = Column(Float, default=0.8)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="metric_configs")
    experiments = relationship("Experiment", back_populates="metric_config")

class MetaLearningConfiguration(Base):
    __tablename__ = "meta_learning_configurations"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    model_type = Column(String, default="xgboost")
    hyperparameters = Column(JSON, default={})
    feature_selection = Column(JSON, default={})
    is_active = Column(Boolean, default=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="meta_learning_configs")

class Experiment(Base):
    __tablename__ = "experiments"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String, default="pending")
    model_config_id = Column(String, ForeignKey("model_configurations.id"), nullable=False)
    metric_config_id = Column(String, ForeignKey("metric_configurations.id"), nullable=False)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    initial_prompt_id = Column(String, ForeignKey("prompts.id"), nullable=False)
    best_prompt_id = Column(String, ForeignKey("prompts.id"), nullable=True)
    max_iterations = Column(Integer, default=5)
    strategy = Column(String, nullable=False)
    iterations_completed = Column(Integer, default=0)
    current_best_score = Column(Float, default=0.0)
    improvement_percentage = Column(Float, default=0.0)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    model_config = relationship("ModelConfiguration", back_populates="experiments")
    metric_config = relationship("MetricConfiguration", back_populates="experiments")
    dataset = relationship("Dataset", back_populates="experiments")
    initial_prompt = relationship("Prompt", foreign_keys=[initial_prompt_id], back_populates="initial_experiments")
    best_prompt = relationship("Prompt", foreign_keys=[best_prompt_id], back_populates="best_experiments")
    user = relationship("User", back_populates="experiments")
    metrics_records = relationship("MetricsRecord", back_populates="experiment")

class MetricsRecord(Base):
    __tablename__ = "metrics_records"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False)
    iteration = Column(Integer, nullable=False)
    metrics = Column(JSON, nullable=False)
    prompt_id = Column(String, ForeignKey("prompts.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="metrics_records")
    prompt = relationship("Prompt", back_populates="metrics_records")
